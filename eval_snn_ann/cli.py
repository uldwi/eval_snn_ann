"""Command-line entry point for model training and export."""
import argparse
import sys

from .data.registry import print_available_datasets_table
from .workflow import train_and_export


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ANN/SNN models, select datasets, optimize models, and export to TFLite/C")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config.yaml",
        help="Path to a YAML or JSON configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default=None,
        help="Select a dataset (e.g. mnist, fashion_mnist, cifar10, speech_commands, physionet_eeg_mi, har_time_series, eeg_data, ecg_data)",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets and domains",
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs",
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=None,
        help="Override the training batch size",
    )
    parser.add_argument(
        "--lr", "--learning-rate",
        type=float,
        default=None,
        dest="learning_rate",
        help="Override the training learning rate",
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        default=None,
        help="Enable early stopping",
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_false",
        dest="early_stopping",
        help="Disable early stopping",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Number of epochs without improvement before stopping",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=None,
        help="Minimum metric improvement required for early stopping",
    )
    parser.add_argument(
        "--monitor",
        type=str,
        default=None,
        choices=["test_acc", "test_loss"],
        help="Metric to monitor: test_acc (default) or test_loss",
    )
    parser.add_argument(
        "-n", "--test-samples",
        type=int,
        default=None,
        dest="test_samples",
        help="Override the number of test samples to export",
    )
    parser.add_argument(
        "--force-train", "--force-retrain",
        dest="force_retrain",
        action="store_true",
        default=False,
        help="Retrain all models even when matching checkpoints exist",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory for saving and loading model checkpoints (default: ./saved_models)",
    )

    # Post-training optimization options
    parser.add_argument(
        "--prune", "--prune-amount",
        type=float,
        default=None,
        dest="prune_amount",
        help="Enable post-training pruning and set the target fraction of removed weights (e.g. 0.3 for 30%% sparsity)",
    )
    parser.add_argument(
        "--prune-method",
        type=str,
        default=None,
        choices=["l1_unstructured", "random_unstructured", "structured", "threshold"],
        help="Pruning method: l1_unstructured (default), random_unstructured, structured, or threshold",
    )
    parser.add_argument(
        "--quantize", "--quantize-type",
        type=str,
        default=None,
        dest="quantize_type",
        choices=["int8", "int4", "fp16", "dynamic"],
        help="Enable post-training weight quantization (int8, int4, fp16, or dynamic)",
    )
    parser.add_argument(
        "--clustering", "--num-clusters",
        type=int,
        default=None,
        dest="num_clusters",
        help="Enable weight clustering with k centroids (e.g. 16 for a 4-bit codebook)",
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=None,
        help="Number of fine-tuning epochs after pruning to recover accuracy",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Training device (cpu, cuda, or mps; default: cpu). Use cuda on DACHS GPU nodes.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        default=False,
        help="Skip host evaluation and TFLite export after training (e.g. when no ESP32/Joulescope is available).",
    )

    args = parser.parse_args()

    if args.list_datasets:
        print_available_datasets_table()
        sys.exit(0)

    train_and_export(
        config_path=args.config,
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        early_stopping=args.early_stopping,
        patience=args.patience,
        min_delta=args.min_delta,
        monitor=args.monitor,
        num_test_samples=args.test_samples,
        force_retrain=args.force_retrain,
        save_dir=args.save_dir,
        prune_amount=args.prune_amount,
        prune_method=args.prune_method,
        quantize_type=args.quantize_type,
        num_clusters=args.num_clusters,
        fine_tune_epochs=args.fine_tune_epochs,
        device=args.device,
        skip_evaluation=args.skip_evaluation,
    )
