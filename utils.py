import importlib.util
import json
import math
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def is_wsl():
    try:
        with open("/proc/version", "r", encoding="utf-8") as handle:
            version_text = handle.read().lower()
        return "microsoft" in version_text or "wsl" in version_text
    except OSError:
        return False


def load_python_config(config_path):
    spec = importlib.util.spec_from_file_location("chromformer_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load config file: {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "CONFIG"):
        config = module.CONFIG
    elif hasattr(module, "config"):
        config = module.config
    else:
        raise AttributeError("Config file must define CONFIG or config.")

    if not isinstance(config, dict):
        raise TypeError("CONFIG must be a dictionary.")

    validate_project_config(config)
    return config


def validate_project_config(config):
    data_cfg = config.get("data", {})
    transform_cfg = data_cfg.get("target_transform", {})
    if not bool(transform_cfg.get("normalize", True)):
        raise ValueError(
            "ChromFormer now only supports NORMALIZE_TARGET=True. "
            "Please update target_transform.normalize to True."
        )
    if data_cfg.get("use_ntv3_gene_embeddings", False):
        raise ValueError(
            "ChromFormer now only supports USE_NTV3=False. "
            "Please remove use_ntv3_gene_embeddings or set it to False."
        )

    required_data_keys = [
        "dataset_name",
        "chromosome",
        "graph_dir",
        "manifest_path",
        "cell_list_path",
        "cell_type_path",
        "gene_table_path",
        "gene_bed_path",
        "label_matrix_path",
        "neighbor_map_path",
    ]
    missing_keys = [key for key in required_data_keys if key not in data_cfg]
    if missing_keys:
        raise KeyError(f"Config data section is missing required keys: {missing_keys}")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def build_output_dirs(output_cfg):
    root_dir = ensure_dir(output_cfg["root_dir"])
    return {
        "root": root_dir,
        "models": ensure_dir(os.path.join(root_dir, output_cfg["model_dir"])),
        "results": ensure_dir(os.path.join(root_dir, output_cfg["result_dir"])),
        "vis": ensure_dir(os.path.join(root_dir, output_cfg["vis_dir"])),
        "logs": ensure_dir(os.path.join(root_dir, output_cfg["log_dir"])),
    }


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def load_cell_order(cell_list_path):
    cell_table = pd.read_csv(cell_list_path, sep="\t", index_col=0)
    return [str(cell_name) for cell_name in cell_table["Cellname"]]


def build_cell_label_map(cell_list_path, cell_type_path):
    cell_order = load_cell_order(cell_list_path)
    label_table = pd.read_csv(cell_type_path, sep="\t", index_col=0)
    label_list = [str(cell_type) for cell_type in label_table["Celltype"]]

    if len(cell_order) != len(label_list):
        raise ValueError(
            "Cell list and cell type files have different lengths: "
            f"{len(cell_order)} vs {len(label_list)}"
        )

    return {cell_id: label for cell_id, label in zip(cell_order, label_list)}


def scan_graph_files(graph_dir, file_suffix):
    graph_dir_path = Path(graph_dir)
    if not graph_dir_path.exists():
        raise FileNotFoundError(f"Graph directory does not exist: {graph_dir}")

    files = sorted(graph_dir_path.glob(f"*{file_suffix}"))
    if not files:
        raise FileNotFoundError(f"No graph files matching '*{file_suffix}' found in {graph_dir}")
    return files


def infer_graph_storage_metadata(graph_path):
    with np.load(graph_path, allow_pickle=False) as payload:
        num_genes = int(payload["num_genes"]) if "num_genes" in payload else int(payload["y"].shape[-1])
        edge_dim = int(payload["edge_attr"].shape[-1])
    return {
        "num_genes": num_genes,
        "edge_dim": edge_dim,
    }


def create_graph_manifest(graph_dir, manifest_path, file_suffix=".npz", chromosome=None):
    files = scan_graph_files(graph_dir, file_suffix)
    first_meta = infer_graph_storage_metadata(str(files[0]))

    samples = []
    total_bytes = 0
    for graph_file in files:
        file_size = graph_file.stat().st_size
        total_bytes += file_size
        samples.append(
            {
                "cell_id": graph_file.stem,
                "file_name": graph_file.name,
                "file_size_bytes": int(file_size),
            }
        )

    manifest = {
        "chromosome": chromosome,
        "graph_dir": str(Path(graph_dir).as_posix()),
        "file_suffix": file_suffix,
        "num_graphs": len(samples),
        "num_genes": first_meta["num_genes"],
        "edge_dim": first_meta["edge_dim"],
        "total_size_bytes": int(total_bytes),
        "samples": samples,
    }
    save_json(manifest_path, manifest)
    return manifest


def load_graph_manifest(data_cfg):
    manifest_path = data_cfg["manifest_path"]
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        configured_graph_dir = str(Path(data_cfg["graph_dir"]).as_posix())
        if manifest.get("graph_dir") != configured_graph_dir:
            manifest["graph_dir"] = configured_graph_dir
            save_json(manifest_path, manifest)
        return manifest

    if not data_cfg.get("rebuild_manifest_if_missing", True):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    ensure_dir(os.path.dirname(manifest_path) or ".")
    return create_graph_manifest(
        graph_dir=data_cfg["graph_dir"],
        manifest_path=manifest_path,
        file_suffix=data_cfg.get("graph_file_suffix", ".npz"),
        chromosome=data_cfg.get("chromosome"),
    )


def build_sample_records(data_cfg):
    manifest = load_graph_manifest(data_cfg)
    cell_order = load_cell_order(data_cfg["cell_list_path"])
    label_map = build_cell_label_map(data_cfg["cell_list_path"], data_cfg["cell_type_path"])

    graph_dir_path = Path(data_cfg["graph_dir"])
    sample_lookup = {
        sample["cell_id"]: graph_dir_path / sample["file_name"]
        for sample in manifest["samples"]
    }

    limit_cells = data_cfg.get("limit_cells")
    sample_records = []
    missing_labels = []

    for cell_id in cell_order:
        graph_path = sample_lookup.get(cell_id)
        if graph_path is None:
            continue
        label = label_map.get(cell_id)
        if label is None:
            missing_labels.append(cell_id)
            continue
        sample_records.append(
            {
                "cell_id": cell_id,
                "label": label,
                "graph_path": str(graph_path),
            }
        )
        if limit_cells is not None and len(sample_records) >= limit_cells:
            break

    if not sample_records:
        raise RuntimeError("No graph samples matched both the manifest and the cell metadata.")

    if missing_labels:
        print(f"Warning: {len(missing_labels)} graph files have no matching cell type labels and were skipped.")

    metadata = {
        "manifest_path": data_cfg["manifest_path"],
        "graph_dir": str(graph_dir_path),
        "num_genes": int(manifest["num_genes"]),
        "edge_dim": int(manifest["edge_dim"]),
        "num_graphs_in_manifest": int(manifest["num_graphs"]),
        "total_records_loaded": len(sample_records),
    }
    return sample_records, metadata


def split_sample_records(sample_records, split_cfg):
    train_types = set(split_cfg["train_types"])
    val_types = set(split_cfg["val_types"])
    test_types = set(split_cfg["test_types"])

    overlaps = {
        "train_val": sorted(train_types & val_types),
        "train_test": sorted(train_types & test_types),
        "val_test": sorted(val_types & test_types),
    }
    duplicated_types = {name: values for name, values in overlaps.items() if values}
    if duplicated_types:
        raise ValueError(f"Split configuration contains overlapping cell types: {duplicated_types}")

    train_records = [record for record in sample_records if record["label"] in train_types]
    val_records = [record for record in sample_records if record["label"] in val_types]
    test_records = [record for record in sample_records if record["label"] in test_types]

    return {
        "train_records": train_records,
        "val_records": val_records,
        "test_records": test_records,
        "train_idx": [idx for idx, record in enumerate(sample_records) if record["label"] in train_types],
        "val_idx": [idx for idx, record in enumerate(sample_records) if record["label"] in val_types],
        "test_idx": [idx for idx, record in enumerate(sample_records) if record["label"] in test_types],
        "train_labels": [record["label"] for record in train_records],
        "val_labels": [record["label"] for record in val_records],
        "test_labels": [record["label"] for record in test_records],
    }


def validate_dataset_split(dataset_split):
    if not dataset_split["train_records"]:
        raise ValueError("Train split is empty. Please check train_types in the config.")
    if not dataset_split["val_records"]:
        raise ValueError("Validation split is empty. Please check val_types in the config.")
    if not dataset_split["test_records"]:
        raise ValueError("Test split is empty. Please check test_types in the config.")


def summarize_split(dataset_split, all_records):
    def count_labels(records):
        counts = {}
        for record in records:
            label = record["label"]
            counts[label] = counts.get(label, 0) + 1
        return counts

    used_indices = set(dataset_split["train_idx"]) | set(dataset_split["val_idx"]) | set(dataset_split["test_idx"])
    unused_records = [record for idx, record in enumerate(all_records) if idx not in used_indices]

    return {
        "total_cells": len(all_records),
        "used_cells": len(used_indices),
        "unused_cells": len(all_records) - len(used_indices),
        "train_size": len(dataset_split["train_records"]),
        "val_size": len(dataset_split["val_records"]),
        "test_size": len(dataset_split["test_records"]),
        "unused_label_counts": count_labels(unused_records),
        "train_label_counts": count_labels(dataset_split["train_records"]),
        "val_label_counts": count_labels(dataset_split["val_records"]),
        "test_label_counts": count_labels(dataset_split["test_records"]),
    }


def expand_half_edges(edge_index_half, edge_attr_half):
    src = edge_index_half[0]
    dst = edge_index_half[1]
    non_diag_mask = src != dst

    if non_diag_mask.any():
        reversed_edge_index = torch.stack([dst[non_diag_mask], src[non_diag_mask]], dim=0)
        reversed_edge_attr = edge_attr_half[non_diag_mask]
        edge_index = torch.cat([edge_index_half, reversed_edge_index], dim=1)
        edge_attr = torch.cat([edge_attr_half, reversed_edge_attr], dim=0)
        return edge_index, edge_attr

    return edge_index_half, edge_attr_half


class LazyGraphDataset(Dataset):
    def __init__(self, records, num_genes):
        self.records = records
        self.num_genes = int(num_genes)
        self.base_x = torch.arange(self.num_genes, dtype=torch.long).view(-1, 1)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        with np.load(record["graph_path"], allow_pickle=False) as payload:
            edge_index_half = torch.from_numpy(payload["edge_index"].astype(np.int64, copy=False))
            edge_attr_half = torch.from_numpy(payload["edge_attr"].astype(np.float32, copy=False))
            y = torch.from_numpy(payload["y"].astype(np.float32, copy=False)).view(1, -1)

        edge_index, edge_attr = expand_half_edges(edge_index_half, edge_attr_half)
        return Data(
            x=self.base_x.clone(),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            cell_id=record["cell_id"],
        )


def build_graph_datasets(dataset_split, num_genes):
    return {
        "train_dataset": LazyGraphDataset(dataset_split["train_records"], num_genes),
        "val_dataset": LazyGraphDataset(dataset_split["val_records"], num_genes),
        "test_dataset": LazyGraphDataset(dataset_split["test_records"], num_genes),
    }


def build_dataloaders(graph_datasets, training_cfg, device=None):
    batch_size = training_cfg["batch_size"]
    requested_num_workers = training_cfg.get("num_workers", 0)
    num_workers = requested_num_workers
    drop_last_train = training_cfg.get("drop_last_train", True)
    prefetch_factor = training_cfg.get("prefetch_factor")

    if is_wsl() and requested_num_workers > 0:
        print(
            "WSL environment detected. Falling back to num_workers=0 for DataLoader "
            "to avoid common multiprocessing segmentation faults on mounted drives."
        )
        num_workers = 0

    persistent_workers = training_cfg.get("persistent_workers", False) and num_workers > 0

    pin_memory = False
    if device is not None and str(device).startswith("cuda"):
        pin_memory = training_cfg.get("pin_memory", False)

    common_loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0 and prefetch_factor is not None:
        common_loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        graph_datasets["train_dataset"],
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last_train,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        graph_datasets["val_dataset"],
        batch_size=batch_size,
        shuffle=False,
        **common_loader_kwargs,
    )
    test_loader = DataLoader(
        graph_datasets["test_dataset"],
        batch_size=batch_size,
        shuffle=False,
        **common_loader_kwargs,
    )
    return train_loader, val_loader, test_loader


def transform_expression_targets_torch(values, transform_cfg):
    transformed = torch.log1p(values.float())
    transformed = torch.clamp(transformed, max=transform_cfg["clip_max"])
    transformed = transformed / transform_cfg["scale_max"]
    return transformed


def _to_serializable(value):
    if isinstance(value, dict):
        return {str(key): _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_to_serializable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _to_serializable(value.item())
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    return str(value)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_to_serializable(payload), handle, ensure_ascii=False, indent=2)


def save_pickle(path, payload):
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def save_numpy_array(path, array):
    np.save(path, np.asarray(array))
