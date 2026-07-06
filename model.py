import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch


class ZINBLoss(nn.Module):
    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, x, mean, disp, pi, scale_factor=1.0):
        eps = self.eps

        x = x.float()
        scale_factor = torch.as_tensor(scale_factor, device=x.device, dtype=x.dtype)
        mean = torch.clamp(mean.float() * scale_factor, min=eps, max=1e7)
        disp = torch.clamp(disp.float(), min=eps, max=1e7)
        pi = torch.clamp(pi.float(), min=eps, max=1.0 - eps)

        t1 = torch.lgamma(disp + x + eps) - torch.lgamma(disp + eps) - torch.lgamma(x + 1.0 + eps)
        t2 = (disp + x) * torch.log1p(mean / (disp + eps)) + x * (
            torch.log(disp + eps) - torch.log(mean + eps)
        )
        nb_log_prob = t1 - t2
        nb_zero_prob = torch.pow(disp / (disp + mean + eps), disp)

        zero_case_log = torch.log(pi + (1.0 - pi) * nb_zero_prob + eps)
        non_zero_case_log = torch.log(1.0 - pi + eps) + nb_log_prob
        log_likelihood = torch.where(x <= 1e-8, zero_case_log, non_zero_case_log)
        return -log_likelihood.mean()


class BiologicallyDrivenAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.2, top_k=20, threshold=0.5, edge_dim=1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}).")

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.top_k = top_k
        self.threshold = threshold

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_bias_proj = nn.Linear(edge_dim, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_bias_map, padding_mask=None, adj_mask=None):
        batch_size, num_nodes, _ = x.size()

        q = self.q_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask_fill_value = -1e4 if scores.dtype in (torch.float16, torch.bfloat16) else -1e9

        if edge_bias_map is not None:
            edge_bias = self.edge_bias_proj(edge_bias_map).permute(0, 3, 1, 2)
            scores = scores + edge_bias

        if adj_mask is not None:
            scores = scores.masked_fill(adj_mask.unsqueeze(1) == 0, mask_fill_value)

        if padding_mask is not None:
            key_mask = padding_mask.view(batch_size, 1, 1, num_nodes)
            scores = scores.masked_fill(~key_mask, mask_fill_value)

        attn_probs = torch.sigmoid(scores)
        threshold_mask = (attn_probs > self.threshold).float()

        k_val = min(max(self.top_k, 1), num_nodes)
        _, topk_indices = torch.topk(attn_probs, k=k_val, dim=-1)
        topk_mask = torch.zeros_like(attn_probs)
        topk_mask.scatter_(-1, topk_indices, 1.0)

        pruned_mask = threshold_mask * topk_mask
        valid_counts = pruned_mask.sum(dim=-1, keepdim=True)
        final_mask = torch.where(valid_counts > 0, pruned_mask, topk_mask)
        scores = scores.masked_fill(final_mask == 0, mask_fill_value)

        attn_weights = F.softmax(scores, dim=-1)
        attention_to_return = attn_weights.detach()
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, num_nodes, -1)
        out = self.out_proj(out)
        out = self.norm(out + x)
        return out, attention_to_return


class ChromFormerPredictor(nn.Module):
    def __init__(
        self,
        num_nodes,
        edge_dim=24,
        hidden_dim=128,
        heads=1,
        dropout=0.2,
        top_k=10,
        threshold=0.5,
    ):
        super().__init__()
        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)

        self.layer1 = BiologicallyDrivenAttention(
            hidden_dim=hidden_dim,
            num_heads=heads,
            dropout=dropout,
            top_k=top_k,
            threshold=threshold,
            edge_dim=edge_dim,
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.head_mu = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ELU(), nn.Linear(64, 1), nn.Softplus())
        self.head_theta = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ELU(), nn.Linear(64, 1), nn.Softplus())
        self.head_pi = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ELU(), nn.Linear(64, 1), nn.Sigmoid())
        self.head_log = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def _node_ids_from_x(self, x):
        if x.dim() == 1:
            return x.long()
        if x.dim() == 2 and x.size(-1) == 1:
            return x.view(-1).long()
        return x[..., 0].contiguous().view(-1).long()

    def forward(self, x, edge_index, edge_attr, batch_idx, num_nodes_in_batch=None):
        node_ids = self._node_ids_from_x(x)
        h = self.node_embedding(node_ids)

        h_dense, padding_mask = to_dense_batch(h, batch_idx)
        edge_bias_dense = to_dense_adj(
            edge_index,
            batch_idx,
            edge_attr=edge_attr,
            max_num_nodes=h_dense.size(1),
        )

        ones_attr = torch.ones(edge_index.size(1), device=edge_index.device)
        adj_mask = to_dense_adj(
            edge_index,
            batch_idx,
            edge_attr=ones_attr,
            max_num_nodes=h_dense.size(1),
        )
        adj_mask = (adj_mask > 0).float()

        batch_size, num_nodes, _ = adj_mask.size()
        eye = torch.eye(num_nodes, device=adj_mask.device).unsqueeze(0).expand(batch_size, -1, -1)
        adj_mask = (adj_mask + eye).clamp(0, 1)

        h_dense, attn1 = self.layer1(h_dense, edge_bias_dense, padding_mask=padding_mask, adj_mask=adj_mask)
        h_dense = self.ffn(h_dense)
        h_out = h_dense[padding_mask]

        mu = self.head_mu(h_out)
        theta = self.head_theta(h_out)
        pi = self.head_pi(h_out)
        s_log = self.head_log(h_out)
        return mu, theta, pi, s_log, [attn1]
