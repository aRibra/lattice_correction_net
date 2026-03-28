# cli.py - Command Line Interface with subcommands for lattice correction network
# 
# Usage:
#   python3 main.py generate --n-simulations 1000
#   python3 main.py train --data-dir <path> --epochs 900
#   python3 main.py evaluate --model-dir <path>
#   python3 main.py benchmark --model-dir <path> --type quad_tilt

import argparse
import torch

from constants import Constants as C
from data import gen_data, load_data_from_dir, prepare_data_for_training
from net import build_model, build_model_from_train_dir, train_model
from eval import main_evaluation_block, inference_on_validation_data
from visualization import plot_data_histograms, print_maes_micron
from utils import serialize_minmax_scaler


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
        """
    )
    
    # Add version or common options here if needed
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # =========================================
    # GENERATE subcommand
    # =========================================
    gen_parser = subparsers.add_parser('generate', 
        help='Generate simulation data',
        description='Generate simulation data for training.')
    
    gen_parser.add_argument('--n-simulations', '-n', type=int, default=1000,
                        help='Number of simulations to generate (default: 1000)')
    
    # =========================================
    # TRAIN subcommand
    # =========================================
    train_parser = subparsers.add_parser('train', 
        help='Train a model',
        description='Train a model on simulation data.')
    
    # Data options
    train_parser.add_argument('--data-dir', '-d', type=str, required=True,
                        help='Data directory containing training data (required)')
    
    # Model architecture options
    train_parser.add_argument('--model-arch', '-m', type=str,
                        choices=[C.NET_ARCH_LSTM, C.NET_ARCH_SIMPLE_FULLY_CONNECTED, C.NET_ARCH_SIMPLE_CNN],
                        default=C.NET_ARCH_LSTM,
                        help=f'Model architecture to use (default: {C.NET_ARCH_LSTM})')
    
    # Training hyperparameters
    train_parser.add_argument('--epochs', '-e', type=int, default=900,
                        help='Number of training epochs (default: 900)')
    train_parser.add_argument('--batch-size', '-b', type=int, default=16,
                        help='Batch size (default: 16)')
    train_parser.add_argument('--test-size', '-t', type=float, default=0.10,
                        help='Test/validation size fraction (default: 0.10)')
    
    # Learning rate (optional, can be added to net.py train_model)
    train_parser.add_argument('--learning-rate', '-lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    
    # Output directory for trained model
    train_parser.add_argument('--output-dir', '-o', type=str, default='exps',
                        help='Output directory for training results (default: exps)')
    
    # =========================================
    # EVALUATE subcommand
    # =========================================
    eval_parser = subparsers.add_parser('evaluate', 
        help='Evaluate a trained model',
        description='Evaluate a trained model on validation data.')
    
    eval_parser.add_argument('--model-dir', '-m', type=str, required=True,
                        help='Model training directory (required)')
    
    # =========================================
    # BENCHMARK subcommand
    # =========================================
    bench_parser = subparsers.add_parser('benchmark', 
        help='Benchmark a trained model',
        description='Run benchmarks on a trained model.')
    
    bench_parser.add_argument('--model-dir', '-m', type=str, required=True,
                        help='Model training directory (required)')
    
    # Benchmark type
    bench_parser.add_argument('--type', '-t', type=str, 
                        choices=['bpm', 'quad_tilt', 'bpm_shift'],
                        default='quad_tilt',
                        help='Benchmark type (default: quad_tilt)')
    
    # Benchmark parameters - BPM noise
    bench_parser.add_argument('--bpm-noise-range', type=float, nargs=2, default=[0, 100e-6],
                        help='Min/max BPM noise range in meters (default: [0, 100μm])')
    
    # Benchmark parameters - Quad tilt noise
    bench_parser.add_argument('--quad-tilt-noise-range', type=float, nargs=2, default=[0.01, 0.05],
                        help='Min/max quadrupole tilt noise range in mrads (default: [0.01, 0.05] mrads)')
    
    # Benchmark parameters - BPM shift
    bench_parser.add_argument('--shift-range', type=float, nargs=2, default=[-100e-6, 100e-6],
                        help='Min/max BPM shift range in meters (default: ±100μm)')
    bench_parser.add_argument('--x-shift', action='store_true',
                        help='Apply shifts on X-axis in BPM shift benchmark')
    bench_parser.add_argument('--y-shift', action='store_true',
                        help='Apply shifts on Y-axis in BPM shift benchmark')
    
    # Common benchmark parameters
    bench_parser.add_argument('--bins', type=int, default=11,
                        help='Number of bins for noise/shift levels (default: 11)')
    bench_parser.add_argument('--runs', type=int, default=50,
                        help='Number of runs per noise/shift level (default: 50)')
    
    return parser


def get_device():
    """Get the device (CUDA or CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        model_arch=args.model_arch
    )
    
    # Build model
    print(f"Building model with architecture: {args.model_arch}")
    model = build_model(args.model_arch, data_shapes, device)
    print(model)
    
    # Get data sub config for saving
    data_automation = sim_data[C.DATA_KEY_DATA_AUTOMATION]
    data_sub_cfg = {    
        'merged_config': sim_data[C.DATA_KEY_MERGED_CONFIG],
        'input_scaler_config': serialize_minmax_scaler(sim_data[C.DATA_KEY_DATASET_SCALERS]['input_scaler']),
        'target_scaler_config': serialize_minmax_scaler(sim_data[C.DATA_KEY_DATASET_SCALERS]['target_scaler']),            
        'overridden_base_config': data_automation.overridden_base_config.copy()
    }
    
    # Train model
    print(f"Training for {args.epochs} epochs...")
    train_results = train_model(
        model, train_loader, val_loader, device, data_sub_cfg, 
        num_epochs=args.epochs
    )
    
    print_maes_micron(
        train_results['val_maes'], 
        sim_data[C.DATA_KEY_DATASET_SCALERS][C.DATA_KEY_TARGET_SCALER]
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
    
    # Load data from the same directory's config
    # Extract data directory from model path or use default
    data_dir = os.path.join(os.path.dirname(os.path.dirname(args.model_dir)), 'data')
    
    print("Loading data for evaluation...")
    if os.path.exists(data_dir):
        sim_data = load_data_from_dir(data_dir)
    else:
        print("Warning: Could not find data directory. Running inference only.")
        sim_data = None
    
    if sim_data is not None:
        # Prepare data
        # Get model architecture from the loaded model config
        model_arch = data_sub_cfg.get('merged_config', {}).get('model_arch', C.NET_ARCH_LSTM)
        
        train_loader, val_loader, _ = prepare_data_for_training(
            sim_data, 
            test_size=0.10, 
            batch_size=16, 
            model_arch=model_arch
        )
        
        # Run inference
        print("Running inference on validation data...")
        inference_on_validation_data(
            model=model, 
            val_loader=val_loader,
            dataset_scalers=sim_data[C.DATA_KEY_DATASET_SCALERS],
            merged_config=data_sub_cfg['merged_config']
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
    
    # Load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(args.model_dir)), 'data')
    
    print("Loading data for benchmark...")
    if os.path.exists(data_dir):
        sim_data = load_data_from_dir(data_dir)
    else:
        print("Error: Could not find data directory for benchmark.")
        return 1
    
    # Prepare data
    model_arch = data_sub_cfg.get('merged_config', {}).get('model_arch', C.NET_ARCH_LSTM)
    
    train_loader, val_loader, _ = prepare_data_for_training(
        sim_data, 
        test_size=0.10, 
        batch_size=16, 
        model_arch=model_arch
    )
    
    # Run benchmark
    print(f"Running {args.type} benchmark...")
    main_evaluation_block(
        model,
        data_sub_cfg,
        val_loader=val_loader,
        benchmark_type=args.type,
        run_benchmark=True,
        bpm_noise_range=args.bpm_noise_range,
        quad_tilt_noise_range=args.quad_tilt_noise_range,
        shift_range=args.shift_range,
        bins=args.bins,
        runs=args.runs,
        x_shift=args.x_shift,
        y_shift=args.y_shift
    )
    
    print("Benchmark completed.")
    return 0


# Mapping of commands to handler functions
COMMAND_HANDLERS = {
    'generate': cmd_generate,
    'train': cmd_train,
    'evaluate': cmd_evaluate,
    'benchmark': cmd_benchmark,
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


if __name__ == '__main__':
    exit(main())
