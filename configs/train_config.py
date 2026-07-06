DATASET_NAME = "brain"  # "embryo" or "brain"
CHROMOSOME = "chr19"
TOP_K = 20


PROJECT_DATA_ROOT = "data"
SHARED_DATA_ROOT = f"{PROJECT_DATA_ROOT}/shared"


DATASET_PRESETS = {
    "embryo": {
        "display_name": "mouse_embryo",
        "dataset_root": f"{PROJECT_DATA_ROOT}/embryo",
        "graph_root": f"{PROJECT_DATA_ROOT}/embryo/metacell_graphs",
        "raw_matrix_root": f"{PROJECT_DATA_ROOT}/embryo/raw_matrices",
        "metadata_dir": f"{PROJECT_DATA_ROOT}/embryo/metadata",
        "label_matrix_dir": f"{PROJECT_DATA_ROOT}/embryo/labels",
        "gene_table_dir": f"{PROJECT_DATA_ROOT}/embryo/genes",
        "gene_bed_path": f"{PROJECT_DATA_ROOT}/embryo/metadata/2_1.emb.gene_in_RNA.bed",
        "splits": {
            "train_types": [
                "blood",
                "mitosis",
                "early neurons",
                "mix late mesenchyme",
                "ExE endoderm",
            ],
            "val_types": [
                "radial glias",
                "early mesoderm",
            ],
            "test_types": [
                "neural ectoderm",
                "early mesenchyme",
                "ExE ectoderm",
            ],
        },
    },
    "brain": {
        "display_name": "mouse_brain",
        "dataset_root": f"{PROJECT_DATA_ROOT}/brain",
        "graph_root": f"{PROJECT_DATA_ROOT}/brain/metacell_graphs",
        "raw_matrix_root": f"{PROJECT_DATA_ROOT}/brain/raw_matrices",
        "metadata_dir": f"{PROJECT_DATA_ROOT}/brain/metadata",
        "label_matrix_dir": f"{PROJECT_DATA_ROOT}/brain/labels",
        "gene_table_dir": f"{PROJECT_DATA_ROOT}/brain/genes",
        "gene_bed_path": f"{PROJECT_DATA_ROOT}/brain/metadata/2_1.brain.gene_in_RNA.bed",
        "rna_source_path": f"{PROJECT_DATA_ROOT}/brain/metadata/GSE223917_HiRES_brain.rna.umicount.tsv",
        "celltype_source_path": f"{PROJECT_DATA_ROOT}/brain/metadata/GSE223917_HiRES_brain_metadata.xlsx",
        "splits": {
            "train_types": [
                "Ex1",
                "In1",
                "Oli",
            ],
            "val_types": [
                "Ex2",
            ],
            "test_types": [
                "Ast",
                "In2",
            ],
        },
    },
}


if DATASET_NAME not in DATASET_PRESETS:
    raise ValueError(f"Unsupported DATASET_NAME: {DATASET_NAME}")

dataset_preset = DATASET_PRESETS[DATASET_NAME]
run_tag = f"{DATASET_NAME}_{CHROMOSOME}_topk{TOP_K}"

CONFIG = {
    "experiment": {
        "name": f"d01_{run_tag}",
        "dataset_name": DATASET_NAME,
    },
    "data": {
        "dataset_name": DATASET_NAME,
        "dataset_display_name": dataset_preset["display_name"],
        "data_root": dataset_preset["dataset_root"],
        "shared_data_root": SHARED_DATA_ROOT,
        "tf_target_path": f"{SHARED_DATA_ROOT}/mouse_core_TF_Target.txt",
        "chromosome": CHROMOSOME,
        "graph_dir": f"{dataset_preset['graph_root']}/{CHROMOSOME}",
        "raw_matrix_dir": f"{dataset_preset['raw_matrix_root']}/{CHROMOSOME}",
        "manifest_path": (
            f"{dataset_preset['metadata_dir']}/"
            f"graph_manifest_{CHROMOSOME}_metacell_global_compressed.json"
        ),
        "graph_file_suffix": ".npz",
        "cell_list_path": f"{dataset_preset['metadata_dir']}/all_cells.csv",
        "cell_type_path": f"{dataset_preset['metadata_dir']}/all_cells_type.csv",
        "gene_table_path": f"{dataset_preset['gene_table_dir']}/all_genes_{CHROMOSOME}.csv",
        "gene_bed_path": dataset_preset["gene_bed_path"],
        "label_matrix_path": f"{dataset_preset['label_matrix_dir']}/all_labels_matrix_{CHROMOSOME}.csv",
        "neighbor_map_path": f"{dataset_preset['metadata_dir']}/cell_neighbors_map_global_{CHROMOSOME}.pickle",
        "rna_source_path": dataset_preset.get("rna_source_path"),
        "celltype_source_path": dataset_preset.get("celltype_source_path"),
        "rebuild_manifest_if_missing": True,
        "limit_cells": None,
        "target_transform": {
            "normalize": True,
            "clip_max": 3.0,
            "scale_max": 3.0,
        },
    },
    "splits": dataset_preset["splits"],
    "model": {
        "hidden_dim": 128,
        "heads": 1,
        "dropout": 0.2,
        "top_k": TOP_K,
        "threshold": 0.5,
        "edge_dim": 24,
    },
    "training": {
        "batch_size": 16,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "epochs": 50,
        "mse_weight": 10.0,
        "patience": 5,
        "grad_clip": 1.0,
        "grad_accum_steps": 2,
        "use_amp": True,
        "drop_last_train": True,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": 2,
        "scheduler_mode": "max",
        "scheduler_factor": 0.5,
        "scheduler_patience": 5,
        "scheduler_metric": "target_pcc",
        "selection_metric": "val_target_pcc",
    },
    "runtime": {
        "seed": 42,
        "device": "cuda",
        "deterministic": False,
        "visualize_batches_per_epoch": 1,
        "visualize_max_samples": 4,
        "visualize_roi": 100,
    },
    "output": {
        "root_dir": f"outputs/train_{run_tag}",
        "model_dir": "models",
        "result_dir": "results",
        "vis_dir": "vis_logs",
        "log_dir": "logs",
        "best_checkpoint_name": "best_model.pth",
        "last_checkpoint_name": "last_model.pth",
        "history_file": "training_history.json",
        "metrics_file": "test_metrics.json",
        "predictions_file": "test_predictions.npy",
        "targets_file": "test_targets.npy",
        "predictions_log_file": "test_predictions_log.npy",
        "targets_log_file": "test_targets_log.npy",
        "test_labels_file": "test_labels.pkl",
        "split_summary_file": "split_summary.json",
        "split_indices_file": "split_indices.json",
        "config_snapshot_file": "resolved_config.json",
        "run_summary_file": "run_summary.json",
    },
}
