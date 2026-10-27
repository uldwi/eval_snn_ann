"""Command-line entry point for model training and export."""
import argparse
import sys

from .data.registry import print_available_datasets_table
from .workflow import train_and_export


def main() -> None:
    parser = argparse.ArgumentParser(description="Neuronale Netze (ANN & SNN) Training, Datensatzauswahl, Post-Training-Optimierung & TFLite/C-Export")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config.yaml",
        help="Pfad zur YAML- oder JSON-Konfigurationsdatei (Standard: 'config.yaml')",
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default=None,
        help="Wählt den Datensatz (z.B. 'mnist', 'fashion_mnist', 'cifar10', 'speech_commands', 'physionet_eeg_mi'/'eeg_mi', 'har_time_series', 'eeg_data', 'ecg_data', 'synthetic_time_series')",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Zeigt alle verfügbaren Datensätze und Domänen in einer Tabelle an",
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=None,
        help="Überschreibt die Anzahl der Trainingsepochen",
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=None,
        help="Überschreibt die Batch-Größe für das Training",
    )
    parser.add_argument(
        "--lr", "--learning-rate",
        type=float,
        default=None,
        dest="learning_rate",
        help="Überschreibt die Lernrate für das Training",
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        default=None,
        help="Aktiviert Early Stopping für das Training",
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_false",
        dest="early_stopping",
        help="Deaktiviert Early Stopping für das Training",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Anzahl an Epochen ohne Verbesserung vor vorzeitigem Abbruch",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=None,
        help="Mindestverbesserung der Metrik für Early Stopping",
    )
    parser.add_argument(
        "--monitor",
        type=str,
        default=None,
        choices=["test_acc", "test_loss"],
        help="Zu überwachende Metrik: 'test_acc' (Standard) oder 'test_loss'",
    )
    parser.add_argument(
        "-n", "--test-samples",
        type=int,
        default=None,
        dest="test_samples",
        help="Überschreibt die Anzahl der zu exportierenden Test-Samples",
    )
    parser.add_argument(
        "--force-train", "--force-retrain",
        dest="force_retrain",
        action="store_true",
        default=False,
        help="Erzwingt ein erneutes Training aller Modelle, auch wenn gespeicherte Checkpoints unverändert vorliegen",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Verzeichnis zum automatischen Speichern und Laden trainierter Modell-Checkpoints (Standard: './saved_models')",
    )

    # Post-Training Optimierungs-Optionen
    parser.add_argument(
        "--prune", "--prune-amount",
        type=float,
        default=None,
        dest="prune_amount",
        help="Aktiviert Post-Training Pruning und setzt den Zielanteil entfernter Gewichte (z. B. 0.3 für 30%% Sparsity)",
    )
    parser.add_argument(
        "--prune-method",
        type=str,
        default=None,
        choices=["l1_unstructured", "random_unstructured", "structured", "threshold"],
        help="Pruning-Methode: 'l1_unstructured' (Standard), 'random_unstructured', 'structured' oder 'threshold'",
    )
    parser.add_argument(
        "--quantize", "--quantize-type",
        type=str,
        default=None,
        dest="quantize_type",
        choices=["int8", "int4", "fp16", "dynamic"],
        help="Aktiviert Post-Training Quantisierung der Gewichte ('int8', 'int4', 'fp16', 'dynamic')",
    )
    parser.add_argument(
        "--clustering", "--num-clusters",
        type=int,
        default=None,
        dest="num_clusters",
        help="Aktiviert Weight-Clustering mit k Centroids (z. B. 16 für 4-Bit Codebook)",
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=None,
        help="Anzahl an Fine-Tuning Epochen nach dem Pruning zur Genauigkeitswiederherstellung",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Gerät für das Training ('cpu', 'cuda' oder 'mps', Standard: 'cpu'). Auf DACHS GPU-Cluster wird 'cuda' verwendet.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        default=False,
        help="Host-Evaluation und TFLite-Export nach dem Training überspringen (z. B. ohne Joulescope/ESP32 auf DACHS).",
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
