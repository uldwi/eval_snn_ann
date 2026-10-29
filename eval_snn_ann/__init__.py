"""ANN/SNN evaluation and embedded deployment toolkit."""

from .config import (
    DEFAULT_CONFIG,
    compute_conv_layer_dims,
    load_config,
    parse_conv_layers,
    parse_hidden_layers,
)
from .data import (
    AVAILABLE_DATASETS,
    DATASET_ALIASES,
    DatasetInfo,
    get_dataset,
    list_available_datasets,
    normalize_dataset_name,
    print_available_datasets_table,
)
from .exporting import export_c_files, export_model_entry_to_main, export_pytorch_to_tflite
from .models import (
    CompactSpikingTransformer,
    CompactSpikeDrivenTransformer,
    CompactTransformerANN,
    ConfigurableANN,
    ConfigurableSpikingJellySNN,
    SpikCommander,
    build_model_from_config,
    encode_input,
    get_encoding_code,
    transfer_ann_to_snn,
)
from .optimization import (
    apply_pruning,
    apply_quantization,
    apply_weight_clustering,
    compute_model_sparsity,
    evaluate_model,
    get_model_optimization_config,
    optimize_model,
)
from .training import compute_model_signature, evaluate_subset, train_model
from .workflow import get_model_configs, train_all_models, train_and_export

__all__ = [
    "AVAILABLE_DATASETS", "DATASET_ALIASES", "DEFAULT_CONFIG", "DatasetInfo",
    "CompactSpikingTransformer", "CompactSpikeDrivenTransformer",
    "CompactTransformerANN", "ConfigurableANN",
    "ConfigurableSpikingJellySNN", "SpikCommander", "apply_pruning",
    "apply_quantization", "apply_weight_clustering", "build_model_from_config",
    "compute_conv_layer_dims", "compute_model_signature", "compute_model_sparsity",
    "encode_input", "evaluate_model", "evaluate_subset", "export_c_files",
    "export_model_entry_to_main", "export_pytorch_to_tflite", "get_dataset",
    "get_encoding_code", "get_model_configs", "get_model_optimization_config",
    "list_available_datasets", "load_config", "normalize_dataset_name",
    "optimize_model", "parse_conv_layers", "parse_hidden_layers",
    "print_available_datasets_table", "train_all_models", "train_and_export",
    "train_model", "transfer_ann_to_snn",
]
