import argparse
import math
import os

import numpy as np
import torch
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
    split_sample_records,
    transform_expression_targets_torch,
    validate_dataset_split,
)
# Keep this import order consistent with train.py to avoid a PyG import-order
# related segmentation fault observed under WSL.
from model import ChromFormerPredictor


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
        "test_target_pcc": pcc,
        "test_target_scc": scc,
        "test_target_mae": mae,
        "test_target_mse": mse,
        "test_target_rmse": rmse,
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


def load_checkpoint_into_model(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint)
    return model


@torch.no_grad()
def evaluate(model, loader, device, target_transform_cfg):
    model.eval()
    preds_raw = []
    targets_raw = []
    preds_target = []
    targets_target = []

    for batch in tqdm(loader, desc="Testing", leave=False):
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


def main(config_path, checkpoint_path=None):
    config = load_python_config(config_path)
    output_dirs = build_output_dirs(config["output"])
    device = get_device(config["runtime"].get("device", "cuda"))
    print(f"Using device: {device}")

    sample_records, graph_metadata = build_sample_records(config["data"])
    dataset_split = split_sample_records(sample_records, config["splits"])
    validate_dataset_split(dataset_split)

    datasets = build_graph_datasets(dataset_split, graph_metadata["num_genes"])
    _, _, test_loader = build_dataloaders(datasets, config["training"], device=device)
    model = build_model(config, graph_metadata["num_genes"], graph_metadata["edge_dim"]).to(device)

    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            output_dirs["models"],
            config["output"]["best_checkpoint_name"],
        )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    load_checkpoint_into_model(model, checkpoint_path, device)
    test_output = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        target_transform_cfg=config["data"]["target_transform"],
    )
    test_metrics = {
        "test_target_pcc": float(test_output["metrics"]["test_target_pcc"]),
        "test_target_scc": float(test_output["metrics"]["test_target_scc"]),
        "test_target_mae": float(test_output["metrics"]["test_target_mae"]),
        "test_target_mse": float(test_output["metrics"]["test_target_mse"]),
        "test_target_rmse": float(test_output["metrics"]["test_target_rmse"]),
    }
    save_inference_artifacts(output_dirs, config["output"], dataset_split, test_output, test_metrics)

    print("Inference complete.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"test_target_pcc: {test_metrics['test_target_pcc']:.6f}")
    print(f"test_target_scc: {test_metrics['test_target_scc']:.6f}")
    print(f"test_target_mae: {test_metrics['test_target_mae']:.6f}")
    print(f"test_target_mse: {test_metrics['test_target_mse']:.6f}")
    print(f"test_target_rmse: {test_metrics['test_target_rmse']:.6f}")
    print(f"Results saved to: {output_dirs['results']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a saved ChromFormer checkpoint.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("configs", "train_config.py"),
        help="Path to the Python config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path. Defaults to the best checkpoint under the configured output directory.",
    )
    args = parser.parse_args()
    main(args.config, args.checkpoint)
