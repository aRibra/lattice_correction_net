# main.py is the main entry point for [generate data >> train model >> evaluate >> benchmark].

import torch
from pylab import *

from net import build_model, build_model_from_train_dir, train_model
from visualization import plot_data_histograms, print_maes_micron
from constants import Constants as C

from data import gen_data, load_data_from_dir, prepare_data_for_training
from eval import main_evaluation_block, inference_on_validation_data
from utils import serialize_minmax_scaler
from sim_config import SHOW_PLOTS

if not SHOW_PLOTS:
    matplotlib.use('Agg')


if __name__ == '__main__':
    
    # -------------------------------
    # Argument Parser Integration
    # -------------------------------
    import argparse
    parser = argparse.ArgumentParser(
        description="Main entry point for generate data >> train model >> evaluate >> benchmark."
    )
    parser.add_argument('--generate-data', action='store_true', default=False,
                        help='Generate data.')
    parser.add_argument('--load-data', action='store_true', default=False,
                        help='Load data from DATA_DIR.')
    parser.add_argument('--prepare-data', action='store_true', default=False,
                        help='Prepare data for training.')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Run training.')
    parser.add_argument('--load-checkpoint', action='store_true', default=True,
                        help='Load model checkpoint for evaluation.')
    parser.add_argument('--evaluate', action='store_true', default=True,
                        help='Run evaluation.')
    parser.add_argument('--benchmark', action='store_true', default=True,
                        help='Run benchmark.')
    parser.add_argument('--n-simulations', type=int, default=1000,
                        help='Number of simulations to generate.')
    parser.add_argument('--data-dir', type=str, default='',
                        help='Data directory for loading data.')
    parser.add_argument('--model-train-dir', type=str,
                        default='exps/exp_LSTM_2000_mix/GOLDEN_run_2025-01-17_00-19-35/training/train_2025-01-17_03-16-42',
                        help='Model training directory.')
    parser.add_argument('--model-arch', type=str, default='LSTM',
                        choices=[C.NET_ARCH_LSTM, C.NET_ARCH_SIMPLE_FULLY_CONNECTED, C.NET_ARCH_SIMPLE_CNN],
                        default=C.NET_ARCH_LSTM,
                        help='Model architecture to use.')
    parser.add_argument('--num-epochs', type=int, default=900,
                        help='Number of training epochs.')
    parser.add_argument('--test-size', type=float, default=0.10,
                        help='Test size fraction.')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size.')
    parser.add_argument('--benchmark-type', type=str, default='quad_tilt',
                        choices=['bpm', 'quad_tilt'],
                        help='Benchmark type.')
    args = parser.parse_args()
    
    # Override mode flags and related parameters with command line arguments
    GENERATE_DATA = args.generate_data
    LOAD_DATA = args.load_data
    PREPARE_DATA = args.prepare_data
    RUN_TRAINING = args.train
    LOAD_CHECKPOINT = args.load_checkpoint
    RUN_EVALUATION = args.evaluate
    RUN_BENCHMARK = args.benchmark
    N_SIMULATIONS = args.n_simulations
    DATA_DIR = args.data_dir
    MODEL_TRAIN_DIR = args.model_train_dir

    if GENERATE_DATA and LOAD_DATA:
        raise Exception("Cannot `GENERATE_DATA` and `LOAD_DATA` at the same time. Choose one.")
    
    if not GENERATE_DATA and not LOAD_DATA and not LOAD_CHECKPOINT:
        raise Exception("To train model choose either `GENERATE_DATA` or `LOAD_DATA`\n\t"
                        "or, choose `LOAD_CHECKPOINT` to load checkpoint and evaluate te model.")
    
    if RUN_TRAINING and LOAD_CHECKPOINT:
        raise Exception("Cannot `RUN_TRAINING` and `LOAD_CHECKPOINT` at the same time. Choose one.")
    
    if not RUN_TRAINING and not LOAD_CHECKPOINT:
        raise Exception("Choose either `RUN_TRAINING` or `LOAD_CHECKPOINT`.")
    
    if not GENERATE_DATA and not LOAD_DATA:
        print("[WARNING]: No data will be loaded or generated for training.")
        PREPARE_DATA = False
        
    
    # -------------------------------
    # Generate or Load data from directory
    # -------------------------------
    sim_data = None
    data_sub_cfg = None
    
    # DATA_DIR = 'data/Sim3000_2000turns_10parts_FODOErr-123457-126-017_avgTrue_tgtquad_misalign_deltas_2'
    # DATA_DIR = 'data/Sim1000_2000turns_10parts_FODOErr-123457--_avgTrue_tgtquad_misalign_deltas_1'
    # DATA_DIR = 'data/Sim2000_6000turns_10parts_FODOErr-123457-136-_avgTrue_tgtquad_misalign_deltas_1'
    
    if GENERATE_DATA:
        sim_data = gen_data(n_simulations=N_SIMULATIONS)

    if LOAD_DATA:
        sim_data = load_data_from_dir(DATA_DIR)
    
    # -------------------------------
    # Plot data histograms
    # -------------------------------
    if sim_data:
        # plot data histograms
        plot_data_histograms(sim_data)
        
        data_automation = sim_data[C.DATA_KEY_DATA_AUTOMATION]
        
        data_sub_cfg = {    
            'merged_config': sim_data[C.DATA_KEY_MERGED_CONFIG],
            'input_scaler_config': serialize_minmax_scaler(sim_data[C.DATA_KEY_DATASET_SCALERS]['input_scaler']),
            'target_scaler_config': serialize_minmax_scaler(sim_data[C.DATA_KEY_DATASET_SCALERS]['target_scaler']),            
            'overridden_base_config': data_automation.overridden_base_config.copy()
        }
    
    # -------------------------------
    # Mdeol Architecture + Device
    # -------------------------------
    # C.NET_ARCH_LSTM | C.NET_ARCH_SIMPLE_FULLY_CONNECTED | C.NET_ARCH_SIMPLE_CNN
    MODEL_ARCH = args.model_arch
    
    # Set model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # -------------------------------
    # Prepare data for training
    # -------------------------------
    if RUN_TRAINING or PREPARE_DATA:
        PREPARE_DATA = True
        batch_size = args.batch_size
        test_size = args.test_size
        train_loader, val_loader, data_shapes = prepare_data_for_training(sim_data, 
                                                                          test_size=test_size, 
                                                                          batch_size=batch_size, 
                                                                          model_arch=MODEL_ARCH)
    
    # -------------------------------
    # Build model from data shapes
    # or, build from config
    # -------------------------------
    if RUN_TRAINING:
        model = build_model(MODEL_ARCH, data_shapes, device)
        
    elif LOAD_CHECKPOINT:
        # Quads MisAligh only
        # MODEL_TRAIN_DIR = "exps/exp_FCNN_1500/run_2025-01-13_12-01-05/training/train_2025-01-13_12-02-58"
        # MODEL_TRAIN_DIR = 'exps/exp_FCNN_1500/run_2025-01-13_21-29-55/training/train_2025-01-13_21-31-18'
        
        # (agumented) Mixed with Quads and Dipoles tilt errors
        # MODEL_TRAIN_DIR = 'exps/exp_Mix_LSTM_3000/training/train_2025-01-13_06-31-50'
        
        # Quads MisAligh only
        # MODEL_TRAIN_DIR = 'exps/exp_FCNN_1000/GOLDEN_run_2025-01-14_08-53-50/training/train_2025-01-14_09-32-33'
        
        # (agumented) 2000 sample / 6000 turns / Mixed with Quads tilt errors
        # MODEL_TRAIN_DIR = 'exps/exp_LSTM_2000_mix/GOLDEN_run_2025-01-17_00-19-35/training/train_2025-01-17_03-16-42'
        
        model, data_sub_cfg = build_model_from_train_dir(MODEL_TRAIN_DIR, device)
    
    print(model)

    # -------------------------------
    # Train model
    # -------------------------------
    # Load checkpoint state_dict
    if RUN_TRAINING:
        num_epochs = args.num_epochs
        train_results = train_model(model, train_loader, val_loader, device, data_sub_cfg, num_epochs=num_epochs)
        print_maes_micron(train_results['val_maes'], sim_data[C.DATA_KEY_DATASET_SCALERS][C.DATA_KEY_TARGET_SCALER])


    # -------------------------------
    # Run Inference on Validation Data
    # -------------------------------
    if PREPARE_DATA:
        inference_on_validation_data(model=model, 
                                 val_loader=val_loader,
                                 dataset_scalers=sim_data[C.DATA_KEY_DATASET_SCALERS],
                                 merged_config=data_sub_cfg['merged_config'])
    
    # -------------------------------
    # Evaluate model
    # -------------------------------
    if RUN_EVALUATION:
        BENCHMARK_TYPE = args.benchmark_type  # 'quad_tilt' # 'bpm' | 'quad_tilt''
        main_evaluation_block(model, data_sub_cfg, benchmark_type=BENCHMARK_TYPE, run_benchmark=RUN_BENCHMARK)

    print("Done.")
    
    # -------------------------------
