import argparse
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from utils import (
    build_dataloaders,
    build_graph_datasets,
    build_output_dirs,
    build_sample_records,
    get_device,
    load_python_config,
    save_json,
    save_numpy_array,
    save_pickle,
    set_seed,
    split_sample_records,
    summarize_split,
    transform_expression_targets_torch,
    validate_dataset_split,
)
# In this WSL + PyG environment, importing utils before model avoids
# a low-level torch_geometric-related segmentation fault during module import.
from model import ChromFormerPredictor, ZINBLoss


def _autocast_context(device, enabled):
    if str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)
    return torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False)


def safe_statistic(fn, y_true, y_pred):
    try:
        value = fn(y_true, y_pred)[0]
    except Exception:
        value = np.nan
    if value is None or not np.isfinite(value):
        return 0.0
    return float(value)


def compute_target_metrics(targets, preds):
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    preds = np.asarray(preds, dtype=np.float64).reshape(-1)

    mae = float(np.mean(np.abs(preds - targets)))
    mse = float(np.mean((preds - targets) ** 2))
    rmse = float(math.sqrt(mse))
    pcc = safe_statistic(pearsonr, targets, preds)
    scc = safe_statistic(spearmanr, targets, preds)

    return {
        "target_pcc": pcc,
        "target_scc": scc,
        "target_mae": mae,
        "target_mse": mse,
        "target_rmse": rmse,
    }


def build_test_metrics_payload(metrics):
    return {
        "test_target_pcc": float(metrics["target_pcc"]),
        "test_target_scc": float(metrics["target_scc"]),
        "test_target_mae": float(metrics["target_mae"]),
        "test_target_mse": float(metrics["target_mse"]),
        "test_target_rmse": float(metrics["target_rmse"]),
    }


def build_model(config, num_genes, edge_dim):
    model_cfg = config["model"]
    return ChromFormerPredictor(
        hidden_dim=model_cfg["hidden_dim"],
        heads=model_cfg["heads"],
        dropout=model_cfg["dropout"],
        top_k=model_cfg["top_k"],
        threshold=model_cfg["threshold"],
        num_nodes=num_genes,
        edge_dim=edge_dim,
    )


def get_grad_scaler(device, enabled):
    if not enabled or not str(device).startswith("cuda"):
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=True)


def load_checkpoint_into_model(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint)
    return model


def train_one_epoch(
    model,
    train_loader,
    criterion_zinb,
    optimizer,
    device,
    mse_weight,
    grad_clip,
    grad_accum_steps,
    use_amp,
    scaler,
    target_transform_cfg,
):
    model.train()
    loss_sum = 0.0
    loss_zinb_sum = 0.0
    loss_mse_sum = 0.0
    optimizer.zero_grad(set_to_none=True)
    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for step_idx, batch in enumerate(progress_bar, start=1):
        batch = batch.to(device, non_blocking=True)

        with _autocast_context(device, use_amp):
            mu, theta, pi, s_pred, _ = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
                batch.x.size(0),
            )
            y_true = batch.y.view(-1, 1).float()
            y_target = transform_expression_targets_torch(y_true, target_transform_cfg)

            loss_zinb = criterion_zinb(y_true, mu, theta, pi)
            loss_mse = F.mse_loss(s_pred, y_target)
            loss = loss_zinb + mse_weight * loss_mse

        loss_to_backprop = loss / grad_accum_steps
        if scaler is not None:
            scaler.scale(loss_to_backprop).backward()
        else:
            loss_to_backprop.backward()

        should_step = step_idx % grad_accum_steps == 0 or step_idx == len(train_loader)
        if should_step:
            if grad_clip is not None and grad_clip > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = batch.num_graphs
        loss_sum += float(loss.item()) * batch_size
        loss_zinb_sum += float(loss_zinb.item()) * batch_size
        loss_mse_sum += float(loss_mse.item()) * batch_size

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            zinb=f"{loss_zinb.item():.4f}",
            mse=f"{loss_mse.item():.4f}",
        )

    return {
        "loss": loss_sum,
        "loss_zinb": loss_zinb_sum,
        "loss_mse": loss_mse_sum,
    }


@torch.no_grad()
def evaluate(model, loader, device, target_transform_cfg, progress_desc="Evaluating"):
    model.eval()
    preds_raw = []
    targets_raw = []
    preds_target = []
    targets_target = []

    for batch in tqdm(loader, desc=progress_desc, leave=False):
        batch = batch.to(device, non_blocking=True)
        mu, theta, pi, s_pred, _ = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.x.size(0),
        )
        del theta, pi

        y_true = batch.y.view(-1, 1).float()
        y_target = transform_expression_targets_torch(y_true, target_transform_cfg)

        preds_raw.append(mu.detach().cpu().numpy().reshape(batch.num_graphs, -1))
        targets_raw.append(y_true.detach().cpu().numpy().reshape(batch.num_graphs, -1))
        preds_target.append(s_pred.detach().cpu().numpy().reshape(batch.num_graphs, -1))
        targets_target.append(y_target.detach().cpu().numpy().reshape(batch.num_graphs, -1))

    preds_raw = np.concatenate(preds_raw, axis=0)
    targets_raw = np.concatenate(targets_raw, axis=0)
    preds_target = np.concatenate(preds_target, axis=0)
    targets_target = np.concatenate(targets_target, axis=0)

    metrics = compute_target_metrics(targets_target, preds_target)
    return {
        "metrics": metrics,
        "preds_raw": preds_raw,
        "targets_raw": targets_raw,
        "preds_target": preds_target,
        "targets_target": targets_target,
    }


def save_inference_artifacts(output_dirs, output_cfg, dataset_split, evaluation_output, metrics):
    save_json(
        os.path.join(output_dirs["results"], output_cfg["metrics_file"]),
        metrics,
    )
    save_numpy_array(
        os.path.join(output_dirs["results"], output_cfg["predictions_file"]),
        evaluation_output["preds_raw"],
    )
    save_numpy_array(
        os.path.join(output_dirs["results"], output_cfg["targets_file"]),
        evaluation_output["targets_raw"],
    )
    save_numpy_array(
        os.path.join(output_dirs["results"], output_cfg["predictions_log_file"]),
        evaluation_output["preds_target"],
    )
    save_numpy_array(
        os.path.join(output_dirs["results"], output_cfg["targets_log_file"]),
        evaluation_output["targets_target"],
    )
    save_pickle(
        os.path.join(output_dirs["results"], output_cfg["test_labels_file"]),
        dataset_split["test_labels"],
    )


def main(config_path):
    config = load_python_config(config_path)
    runtime_cfg = config["runtime"]
    training_cfg = config["training"]
    output_cfg = config["output"]

    set_seed(runtime_cfg["seed"], deterministic=runtime_cfg.get("deterministic", False))
    device = get_device(runtime_cfg.get("device", "cuda"))
    print(f"Using device: {device}")

    output_dirs = build_output_dirs(output_cfg)
    save_json(
        os.path.join(output_dirs["logs"], output_cfg["config_snapshot_file"]),
        config,
    )

    sample_records, graph_metadata = build_sample_records(config["data"])
    dataset_split = split_sample_records(sample_records, config["splits"])
    validate_dataset_split(dataset_split)

    split_summary = summarize_split(dataset_split, sample_records)
    save_json(
        os.path.join(output_dirs["results"], output_cfg["split_summary_file"]),
        split_summary,
    )
    save_json(
        os.path.join(output_dirs["results"], output_cfg["split_indices_file"]),
        {
            "train_cell_ids": [item["cell_id"] for item in dataset_split["train_records"]],
            "val_cell_ids": [item["cell_id"] for item in dataset_split["val_records"]],
            "test_cell_ids": [item["cell_id"] for item in dataset_split["test_records"]],
        },
    )

    print(f"Loading graph manifest from {config['data']['manifest_path']}...")
    print(
        f"Num genes: {graph_metadata['num_genes']}, "
        f"Total cells: {graph_metadata['total_records_loaded']}"
    )
    print(
        f"Split: Train={len(dataset_split['train_records'])}, "
        f"Val={len(dataset_split['val_records'])}, "
        f"Test={len(dataset_split['test_records'])}"
    )

    datasets = build_graph_datasets(dataset_split, graph_metadata["num_genes"])
    train_loader, val_loader, test_loader = build_dataloaders(datasets, training_cfg, device=device)

    model = build_model(config, graph_metadata["num_genes"], graph_metadata["edge_dim"]).to(device)
    criterion_zinb = ZINBLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=training_cfg.get("scheduler_mode", "max"),
        factor=training_cfg.get("scheduler_factor", 0.5),
        patience=training_cfg.get("scheduler_patience", 5),
    )
    scaler = get_grad_scaler(device, training_cfg.get("use_amp", True))
    target_transform_cfg = config["data"]["target_transform"]

    history = []
    best_val_pcc = -float("inf")
    best_epoch = 0
    patience_counter = 0
    best_checkpoint_path = os.path.join(output_dirs["models"], output_cfg["best_checkpoint_name"])
    last_checkpoint_path = os.path.join(output_dirs["models"], output_cfg["last_checkpoint_name"])

    print("Start training ChromFormer...")
    for epoch in range(1, training_cfg["epochs"] + 1):
        train_stats = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion_zinb=criterion_zinb,
            optimizer=optimizer,
            device=device,
            mse_weight=training_cfg["mse_weight"],
            grad_clip=training_cfg.get("grad_clip"),
            grad_accum_steps=training_cfg.get("grad_accum_steps", 1),
            use_amp=training_cfg.get("use_amp", True),
            scaler=scaler,
            target_transform_cfg=target_transform_cfg,
        )
        val_output = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            target_transform_cfg=target_transform_cfg,
            progress_desc="Validation",
        )
        val_metrics = val_output["metrics"]
        scheduler.step(val_metrics["target_pcc"])

        train_size = max(len(dataset_split["train_records"]), 1)
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_stats["loss"] / train_size,
            "train_zinb_loss": train_stats["loss_zinb"] / train_size,
            "train_mse_loss": train_stats["loss_mse"] / train_size,
            "val_target_pcc": val_metrics["target_pcc"],
            "val_target_scc": val_metrics["target_scc"],
            "val_target_mae": val_metrics["target_mae"],
            "val_target_mse": val_metrics["target_mse"],
            "val_target_rmse": val_metrics["target_rmse"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch}: "
            f"Total Loss {epoch_record['train_loss']:.4f} "
            f"(ZINB: {epoch_record['train_zinb_loss']:.4f}, "
            f"MSE: {epoch_record['train_mse_loss']:.4f}) | "
            f"Val PCC {val_metrics['target_pcc']:.4f} | "
            f"Val SCC {val_metrics['target_scc']:.4f} | "
            f"Val RMSE {val_metrics['target_rmse']:.4f} | "
            f"LR {epoch_record['learning_rate']}"
        )

        torch.save(model.state_dict(), last_checkpoint_path)
        if val_metrics["target_pcc"] > best_val_pcc:
            best_val_pcc = val_metrics["target_pcc"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"  -> Best model updated: {best_checkpoint_path}")
        else:
            patience_counter += 1

        if patience_counter >= training_cfg["patience"]:
            print("Early stopping triggered.")
            break

    save_json(
        os.path.join(output_dirs["logs"], output_cfg["history_file"]),
        history,
    )

    load_checkpoint_into_model(model, best_checkpoint_path, device)
    test_output = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        target_transform_cfg=target_transform_cfg,
        progress_desc="Testing",
    )
    test_metrics = build_test_metrics_payload(test_output["metrics"])
    save_inference_artifacts(output_dirs, output_cfg, dataset_split, test_output, test_metrics)

    save_json(
        os.path.join(output_dirs["logs"], output_cfg["run_summary_file"]),
        {
            "best_epoch": best_epoch,
            "best_val_target_pcc": best_val_pcc,
            "best_checkpoint_path": best_checkpoint_path,
            "last_checkpoint_path": last_checkpoint_path,
            "metrics_file": os.path.join(output_dirs["results"], output_cfg["metrics_file"]),
        },
    )

    print("Training complete.")
    print(f"Best val_target_pcc: {best_val_pcc:.4f} at epoch {best_epoch}")
    print(
        f"Test PCC {test_metrics['test_target_pcc']:.4f} | "
        f"Test SCC {test_metrics['test_target_scc']:.4f} | "
        f"Test RMSE {test_metrics['test_target_rmse']:.4f}"
    )
    print(f"Best model: {best_checkpoint_path}")
    print(f"Results saved under: {output_cfg['root_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ChromFormer for scHiC to scRNA prediction.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("configs", "train_config.py"),
        help="Path to the Python config file.",
    )
    args = parser.parse_args()
    main(args.config)
