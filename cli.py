# cli.py - Command Line Interface with subcommands for lattice correction network
#
# Usage:
#   python3 main.py generate --n-simulations 1000
#   python3 main.py train --data-dir <path> --epochs 900
#   python3 main.py evaluate --model-dir <path>
#   python3 main.py benchmark --model-dir <path> --type quad_tilt

import argparse
import random
import torch

from constants import Constants as C
from data import (
    gen_data,
    load_data_from_dir,
    prepare_data_for_training,
    get_data_splits,
)
from net import build_model, build_model_from_train_dir, train_model
from eval import main_evaluation_block, inference_on_validation_data
from visualization import (
    plot_data_histograms,
    print_maes_micron,
    plot_benchmark_accumulated_datasets,
)
from sim_config import SAVE_DIR_BENCHMARKS
from utils import serialize_minmax_scaler
from accumulated_training import train_accumulated_datasets


def create_parser():
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="Lattice Correction Network - Generate data, train model, evaluate, and benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 main.py generate --n-simulations 1000
  python3 main.py train --data-dir data/Sim1000_2000turns_10parts_FODOErr-123457--_avgTrue_tgtquad_misalign_deltas_1 --epochs 900
  python3 main.py evaluate --model-dir exps/exp_LSTM_2000_mix/GOLDEN_run_2025-01-17_00-19-35/training/train_2025-01-17_03-16-42
  python3 main.py benchmark --model-dir <path> --type quad_tilt
        """,
    )

    # Add version or common options here if needed
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # =========================================
    # GENERATE subcommand
    # =========================================
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate simulation data",
        description="Generate simulation data for training.",
    )

    gen_parser.add_argument(
        "--n-simulations",
        "-n",
        type=int,
        default=1000,
        help="Number of simulations to generate (default: 1000)",
    )

    # =========================================
    # TRAIN subcommand
    # =========================================
    train_parser = subparsers.add_parser(
        "train", help="Train a model", description="Train a model on simulation data."
    )

    # Data options
    train_parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        required=True,
        help="Data directory containing training data (required)",
    )

    # Model architecture options
    train_parser.add_argument(
        "--model-arch",
        "-m",
        type=str,
        choices=[
            C.NET_ARCH_LSTM,
            C.NET_ARCH_SIMPLE_FULLY_CONNECTED,
            C.NET_ARCH_SIMPLE_CNN,
        ],
        default=C.NET_ARCH_LSTM,
        help=f"Model architecture to use (default: {C.NET_ARCH_LSTM})",
    )

    # Training hyperparameters
    train_parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=900,
        help="Number of training epochs (default: 900)",
    )
    train_parser.add_argument(
        "--batch-size", "-b", type=int, default=16, help="Batch size (default: 16)"
    )
    train_parser.add_argument(
        "--test-size",
        "-t",
        type=float,
        default=0.10,
        help="Test/validation size fraction (default: 0.10)",
    )

    # Learning rate (optional, can be added to net.py train_model)
    train_parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )

    # Output directory for trained model
    train_parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="exps",
        help="Output directory for training results (default: exps)",
    )

    # PINN options
    train_parser.add_argument(
        "--use-pinn",
        action="store_true",
        default=False,
        help="Enable physics-informed loss (PINN)",
    )
    train_parser.add_argument(
        "--pinn-lambda",
        type=float,
        default=0.2,
        help="PINN physics loss weight (default: 0.2)",
    )

    # =========================================
    # EVALUATE subcommand
    # =========================================
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a trained model",
        description="Evaluate a trained model on validation data.",
    )

    eval_parser.add_argument(
        "--model-dir",
        "-m",
        type=str,
        required=True,
        help="Model training directory (required)",
    )

    eval_parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        required=False,
        help="Data directory for evaluation (if different from auto-detected)",
    )

    # Additional error sources for evaluation
    eval_parser.add_argument(
        "--enable-k-errors",
        action="store_true",
        help="Enable k_errors in evaluation",
    )
    eval_parser.add_argument(
        "--k-drift-range",
        type=float,
        nargs=2,
        default=[0.04, 0.04],
        help="K systemic drift fraction range (default: [0.04, 0.04])",
    )
    eval_parser.add_argument(
        "--k-jitter-range",
        type=float,
        nargs=2,
        default=[0.005, 0.01],
        help="K stochastic jitter fraction range (default: [0.005, 0.01])",
    )
    eval_parser.add_argument(
        "--include-quad-tilt",
        action="store_true",
        help="Include quad_tilt errors in evaluation",
    )
    eval_parser.add_argument(
        "--quad-tilt-range",
        type=float,
        nargs=2,
        default=[0.01, 0.05],
        help="Min/max quadrupole tilt angle range in mrads (default: [0.01, 0.05] mrads)",
    )
    eval_parser.add_argument(
        "--include-bpm-noise",
        action="store_true",
        help="Include BPM noise in evaluation",
    )
    eval_parser.add_argument(
        "--bpm-noise-range",
        type=float,
        nargs=2,
        default=[0, 100e-6],
        help="Min/max BPM noise range in meters (default: [0, 100μm])",
    )

    # =========================================
    # BENCHMARK subcommand
    # =========================================
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Benchmark a trained model",
        description="Run benchmarks on a trained model.",
    )

    bench_parser.add_argument(
        "--model-dir",
        "-m",
        type=str,
        required=True,
        help="Model training directory (required)",
    )

    bench_parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        required=False,
        help="Data directory for benchmark (required for simulation-based benchmarks)",
    )

    # Primary benchmark type
    bench_parser.add_argument(
        "--primary-benchmark",
        "-p",
        type=str,
        choices=["k_errors", "bpm_noise", "quad_tilt", "bpm_shift"],
        default="quad_tilt",
        help="Primary benchmark type to sweep (default: quad_tilt)",
    )

    # K errors parameters
    bench_parser.add_argument(
        "--k-drift-range",
        type=float,
        nargs=2,
        default=[0.04, 0.04],
        help="K systemic drift fraction range (default: [0.04, 0.04])",
    )
    bench_parser.add_argument(
        "--k-jitter-range",
        type=float,
        nargs=2,
        default=[0.005, 0.01],
        help="K stochastic jitter fraction range (default: [0.005, 0.01])",
    )

    # BPM noise parameters
    bench_parser.add_argument(
        "--bpm-noise-range",
        type=float,
        nargs=2,
        default=[0, 100e-6],
        help="Min/max BPM noise range in meters (default: [0, 100μm])",
    )

    # Quad tilt parameters
    bench_parser.add_argument(
        "--quad-tilt-range",
        type=float,
        nargs=2,
        default=[0.01, 0.05],
        help="Min/max quadrupole tilt angle range in mrads (default: [0.01, 0.05] mrads)",
    )

    # Additional error sources (composable with primary benchmark)
    bench_parser.add_argument(
        "--include-k-errors",
        action="store_true",
        help="Include k_errors as additional error source",
    )
    bench_parser.add_argument(
        "--include-quad-tilt",
        action="store_true",
        help="Include quad_tilt errors as additional error source",
    )
    bench_parser.add_argument(
        "--include-bpm-noise",
        action="store_true",
        help="Include BPM noise as additional error source",
    )

    # BPM shift parameters (only for bpm_shift benchmark type)
    bench_parser.add_argument(
        "--shift-range",
        type=float,
        nargs=2,
        default=[-100e-6, 100e-6],
        help="Min/max BPM shift range in meters (default: ±100μm)",
    )
    bench_parser.add_argument(
        "--x-shift",
        action="store_true",
        help="Apply shifts on X-axis in BPM shift benchmark",
    )
    bench_parser.add_argument(
        "--y-shift",
        action="store_true",
        help="Apply shifts on Y-axis in BPM shift benchmark",
    )

    # Common benchmark parameters
    bench_parser.add_argument(
        "--bins",
        type=int,
        default=11,
        help="Number of bins for noise/shift levels (default: 11)",
    )
    bench_parser.add_argument(
        "--runs",
        type=int,
        default=50,
        help="Number of runs per noise/shift level (default: 50)",
    )

    # =========================================
    # ACCUMULATED TRAINING subcommand
    # =========================================
    accum_parser = subparsers.add_parser(
        "accumulated",
        help="Run accumulated training benchmark",
        description="Train on progressively larger subsets of data to benchmark scaling.",
    )

    accum_parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        required=True,
        help="Data directory containing training data (required)",
    )

    accum_parser.add_argument(
        "--model-arch",
        "-m",
        type=str,
        choices=[
            C.NET_ARCH_LSTM,
            C.NET_ARCH_SIMPLE_FULLY_CONNECTED,
            C.NET_ARCH_SIMPLE_CNN,
        ],
        default=C.NET_ARCH_LSTM,
        help=f"Model architecture to use (default: {C.NET_ARCH_LSTM})",
    )

    accum_parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=900,
        help="Number of training epochs (default: 900)",
    )
    accum_parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    accum_parser.add_argument(
        "--test-size",
        "-t",
        type=float,
        default=0.10,
        help="Test/validation size fraction (default: 0.10)",
    )

    accum_parser.add_argument(
        "--num-datasets",
        "-n",
        type=int,
        default=10,
        help="Number of accumulated datasets (default: 10)",
    )

    accum_parser.add_argument(
        "--use-pinn",
        action="store_true",
        default=False,
        help="Enable physics-informed loss (PINN)",
    )
    accum_parser.add_argument(
        "--pinn-lambda",
        type=float,
        default=0.2,
        help="PINN physics loss weight (default: 0.2)",
    )

    return parser


def get_device():
    """Get the device (CUDA or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cmd_generate(args):
    """Handle the generate subcommand."""
    print(f"Generating {args.n_simulations} simulations...")
    sim_data = gen_data(n_simulations=args.n_simulations)
    print("Data generated successfully.")
    print("Done.")
    return 0


def cmd_train(args):
    """Handle the train subcommand."""
    import os

    device = get_device()
    print(f"Using device: {device}")

    # Load data
    print(f"Loading data from: {args.data_dir}")
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist.")
        return 1

    sim_data = load_data_from_dir(args.data_dir)

    # Plot data histograms
    plot_data_histograms(sim_data)

    # Prepare data for training
    print("Preparing data for training...")
    train_loader, val_loader, data_shapes = prepare_data_for_training(
        sim_data,
        test_size=args.test_size,
        batch_size=args.batch_size,
        model_arch=args.model_arch,
    )

    # Build model
    print(f"Building model with architecture: {args.model_arch}")
    model = build_model(args.model_arch, data_shapes, device)
    print(model)

    # Get data sub config for saving
    data_automation = sim_data[C.DATA_KEY_DATA_AUTOMATION]
    data_sub_cfg = {
        "merged_config": sim_data[C.DATA_KEY_MERGED_CONFIG],
        "input_scaler_config": serialize_minmax_scaler(
            sim_data[C.DATA_KEY_DATASET_SCALERS]["input_scaler"]
        ),
        "target_scaler_config": serialize_minmax_scaler(
            sim_data[C.DATA_KEY_DATASET_SCALERS]["target_scaler"]
        ),
        "overridden_base_config": data_automation.overridden_base_config.copy(),
    }

    # Train model
    print(f"Training for {args.epochs} epochs...")
    if args.use_pinn:
        data_sub_cfg["merged_config"]["use_pinn"] = True
        data_sub_cfg["merged_config"]["pinn_lambda"] = args.pinn_lambda
    train_results = train_model(
        model,
        train_loader,
        val_loader,
        device,
        data_sub_cfg,
        num_epochs=args.epochs,
        use_pinn=args.use_pinn,
        pinn_lambda=args.pinn_lambda,
    )

    print_maes_micron(
        train_results["val_maes"],
        sim_data[C.DATA_KEY_DATASET_SCALERS][C.DATA_KEY_TARGET_SCALER],
    )

    print("Training completed successfully.")
    return 0


def cmd_evaluate(args):
    """Handle the evaluate subcommand."""
    import os

    device = get_device()
    print(f"Using device: {device}")

    # Load model from checkpoint
    print(f"Loading model from: {args.model_dir}")
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory '{args.model_dir}' does not exist.")
        return 1

    model, data_sub_cfg = build_model_from_train_dir(args.model_dir, device)
    print(model)

    # Check if any error source is enabled for simulation-based evaluation
    use_simulation = (
        args.enable_k_errors or args.include_quad_tilt or args.include_bpm_noise
    )

    if use_simulation:
        # Load data for simulation-based evaluation
        if args.data_dir:
            data_dir = args.data_dir
        else:
            data_dir = os.path.join(
                os.path.dirname(os.path.dirname(args.model_dir)), "data"
            )

        print(f"Loading data from: {data_dir}")
        if not os.path.exists(data_dir):
            print(f"Error: Data directory '{data_dir}' does not exist.")
            return 1

        sim_data = load_data_from_dir(data_dir)

        # Get model architecture from the loaded model config
        model_arch = data_sub_cfg.get("merged_config", {}).get(
            "model_arch", C.NET_ARCH_LSTM
        )

        train_loader, val_loader, _ = prepare_data_for_training(
            sim_data, test_size=0.10, batch_size=16, model_arch=model_arch
        )

        # Run simulation-based evaluation with error sources
        print("Running simulation-based evaluation with error sources...")
        main_evaluation_block(
            model=model,
            data_sub_cfg=data_sub_cfg,
            val_loader=val_loader,
            run_benchmark=False,
            enable_k_errors=args.enable_k_errors,
            k_drift_range=args.k_drift_range,
            k_jitter_range=args.k_jitter_range,
            include_quad_tilt=args.include_quad_tilt,
            quad_tilt_range=args.quad_tilt_range,
            include_bpm_noise=args.include_bpm_noise,
            bpm_noise_range=args.bpm_noise_range,
            data_dir=data_dir,
        )
    else:
        # Load data from the same directory's config
        # Extract data directory from model path or use default
        if args.data_dir:
            data_dir = args.data_dir
        else:
            data_dir = os.path.join(
                os.path.dirname(os.path.dirname(args.model_dir)), "data"
            )

        print(f"Loading data from: {data_dir}")
        if os.path.exists(data_dir):
            sim_data = load_data_from_dir(data_dir)
        else:
            print("Warning: Could not find data directory. Running inference only.")
            sim_data = None

        if sim_data is not None:
            # Prepare data
            # Get model architecture from the loaded model config
            model_arch = data_sub_cfg.get("merged_config", {}).get(
                "model_arch", C.NET_ARCH_LSTM
            )

            train_loader, val_loader, _ = prepare_data_for_training(
                sim_data, test_size=0.10, batch_size=16, model_arch=model_arch
            )

            # Run inference
            print("Running inference on validation data...")
            inference_on_validation_data(
                model=model,
                val_loader=val_loader,
                dataset_scalers=sim_data[C.DATA_KEY_DATASET_SCALERS],
                merged_config=data_sub_cfg["merged_config"],
            )

    print("Evaluation completed.")
    return 0


def cmd_benchmark(args):
    """Handle the benchmark subcommand."""
    import os

    device = get_device()
    print(f"Using device: {device}")

    # Load model from checkpoint
    print(f"Loading model from: {args.model_dir}")
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory '{args.model_dir}' does not exist.")
        return 1

    model, data_sub_cfg = build_model_from_train_dir(args.model_dir, device)
    print(model)

    # Validate additional error sources for bpm_shift
    if args.primary_benchmark == "bpm_shift":
        if args.include_k_errors or args.include_quad_tilt or args.include_bpm_noise:
            print(
                "Warning: bpm_shift benchmark uses existing data. "
                "Disabling additional error sources."
            )
        # bpm_shift doesn't use additional error sources
        enable_k_errors = False
        include_quad_tilt = False
        include_bpm_noise = False
    else:
        enable_k_errors = args.include_k_errors or args.primary_benchmark == "k_errors"
        include_quad_tilt = args.include_quad_tilt
        include_bpm_noise = args.include_bpm_noise

    # Determine data directory
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(args.model_dir)), "data"
        )

    # For simulation-based benchmarks, data_dir is required
    if args.primary_benchmark != "bpm_shift":
        if not os.path.exists(data_dir):
            print(f"Error: Data directory '{data_dir}' does not exist.")
            print("Hint: Use --data-dir to specify the data directory.")
            return 1
        print(f"Loading data from: {data_dir}")
        sim_data = load_data_from_dir(data_dir)

        # Prepare data
        model_arch = data_sub_cfg.get("merged_config", {}).get(
            "model_arch", C.NET_ARCH_LSTM
        )

        train_loader, val_loader, _ = prepare_data_for_training(
            sim_data, test_size=0.10, batch_size=16, model_arch=model_arch
        )
    else:
        # bpm_shift uses existing data
        if os.path.exists(data_dir):
            sim_data = load_data_from_dir(data_dir)
            model_arch = data_sub_cfg.get("merged_config", {}).get(
                "model_arch", C.NET_ARCH_LSTM
            )
            train_loader, val_loader, _ = prepare_data_for_training(
                sim_data, test_size=0.10, batch_size=16, model_arch=model_arch
            )
        else:
            print(
                f"Error: Data directory '{data_dir}' does not exist for bpm_shift benchmark."
            )
            return 1

    # Run benchmark
    print(f"Running {args.primary_benchmark} benchmark...")
    main_evaluation_block(
        model=model,
        data_sub_cfg=data_sub_cfg,
        val_loader=val_loader,
        primary_benchmark=args.primary_benchmark,
        run_benchmark=True,
        bins=args.bins,
        runs=args.runs,
        enable_k_errors=enable_k_errors,
        k_drift_range=args.k_drift_range,
        k_jitter_range=args.k_jitter_range,
        include_quad_tilt=include_quad_tilt,
        quad_tilt_range=args.quad_tilt_range,
        include_bpm_noise=include_bpm_noise,
        bpm_noise_range=args.bpm_noise_range,
        shift_range=args.shift_range,
        x_shift=args.x_shift,
        y_shift=args.y_shift,
        data_dir=data_dir,
    )

    print("Benchmark completed.")
    return 0


def cmd_accumulated_training(args):
    import numpy as np
    torch.manual_seed(42)  # For reproducibility
    torch.cuda.manual_seed(42)
    np.random.seed(42)  # For reproducibility
    random.seed(42)  # For reproducibility

    """Handle the accumulated training benchmark subcommand."""
    import os

    device = get_device()
    print(f"Using device: {device}")

    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist.")
        return 1

    print(f"Loading data from: {args.data_dir}")
    sim_data = load_data_from_dir(args.data_dir)

    data_sub_cfg = {
        "merged_config": sim_data[C.DATA_KEY_MERGED_CONFIG],
        "input_scaler_config": serialize_minmax_scaler(
            sim_data[C.DATA_KEY_DATASET_SCALERS]["input_scaler"]
        ),
        "target_scaler_config": serialize_minmax_scaler(
            sim_data[C.DATA_KEY_DATASET_SCALERS]["target_scaler"]
        ),
        "overridden_base_config": sim_data[
            C.DATA_KEY_DATA_AUTOMATION
        ].overridden_base_config.copy(),
    }

    if args.use_pinn:
        data_sub_cfg["merged_config"]["use_pinn"] = True
        data_sub_cfg["merged_config"]["pinn_lambda"] = args.pinn_lambda

    train_inputs, val_inputs, train_targets, val_targets, data_shapes = get_data_splits(
        sim_data, test_size=args.test_size, model_arch=args.model_arch
    )

    print(
        f"Running accumulated training benchmark with {args.num_datasets} datasets..."
    )
    print(f"  Data directory: {args.data_dir}")
    print(f"  Model architecture: {args.model_arch}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Validation split: {args.test_size}")
    print(f"  PINN enabled: {args.use_pinn}")
    if args.use_pinn:
        print(f"  PINN lambda: {args.pinn_lambda}")

    sample_sizes, accuracies_val, accuracies_train = train_accumulated_datasets(
        X_train=train_inputs,
        y_train=train_targets,
        X_val=val_inputs,
        y_val=val_targets,
        batch_size=args.batch_size,
        data_shapes=data_shapes,
        data_sub_cfg=data_sub_cfg,
        number_of_accumulated_datasets=args.num_datasets,
        model_arch=args.model_arch,
        device=device,
        num_epochs=args.epochs,
        use_pinn=args.use_pinn,
        pinn_lambda=args.pinn_lambda,
    )

    dataset_scalers = sim_data[C.DATA_KEY_DATASET_SCALERS]

    benchmark_results = {
        C.KEY_ACCUMULATED_DATASETS: {
            "nb_datasets": args.num_datasets,
            "results_val_mae": accuracies_val,
            "results_train_mae": accuracies_train,
        }
    }

    benchmark_results[C.KEY_ACCUMULATED_DATASETS]["results_val_mae_unscaled"] = {}
    benchmark_results[C.KEY_ACCUMULATED_DATASETS]["results_train_mae_unscaled"] = {}

    mean_min, mean_max = (
        dataset_scalers["target_scaler"].data_min_.mean(),
        dataset_scalers["target_scaler"].data_max_.mean(),
    )
    mean_scaler = mean_max - mean_min

    print("\nValidation MAE (unscaled):")
    for ii, accc in accuracies_val.items():
        mean_unscaled = accc * mean_scaler
        print(f"{len(accuracies_val)}, {mean_unscaled:.7f}", f"{mean_unscaled * 1e6:.2f}")
        benchmark_results[C.KEY_ACCUMULATED_DATASETS]["results_val_mae_unscaled"][
            ii
        ] = mean_unscaled

    print("\nTraining MAEs (unscaled):")
    for ii, accc in accuracies_train.items():
        mean_unscaled = accc * mean_scaler
        print(f"{len(accuracies_train)}, {mean_unscaled:.7f}", f"{mean_unscaled * 1e6:.2f}")
        benchmark_results[C.KEY_ACCUMULATED_DATASETS]["results_train_mae_unscaled"][
            ii
        ] = mean_unscaled

    torch.save(
        benchmark_results,
        f"{SAVE_DIR_BENCHMARKS}/benchmark_results_accumulated_datasets.pt",
    )

    plot_benchmark_accumulated_datasets(benchmark_results)

    print("Accumulated training benchmark completed.")
    return 0


# Mapping of commands to handler functions
COMMAND_HANDLERS = {
    "generate": cmd_generate,
    "train": cmd_train,
    "evaluate": cmd_evaluate,
    "benchmark": cmd_benchmark,
    "accumulated": cmd_accumulated_training,
}


def main():
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Get the handler for this command
    handler = COMMAND_HANDLERS.get(args.command)
    if handler is None:
        print(f"Error: Unknown command '{args.command}'")
        parser.print_help()
        return 1

    # Execute the command
    return handler(args)


if __name__ == "__main__":
    exit(main())
