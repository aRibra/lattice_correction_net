# accumulated_training.py

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib

from sim_config import SAVE_DIR_BENCHMARKS, SHOW_PLOTS
from net import build_model, train_model
from data import load_data_from_dir, get_data_splits, get_data_sub_cfg

from visualization import plot_benchmark_accumulated_datasets
from constants import Constants as C

if not SHOW_PLOTS:
    matplotlib.use('Agg')


def train_accumulated_datasets(X_train,
                               y_train,
                               X_val,
                               y_val,
                               batch_size,
                               data_shapes,
                               data_sub_cfg,
                               number_of_accumulated_datasets=10, 
                               model_arch=C.NET_ARCH_LSTM, 
                               device='cuda',
                               num_epochs=600):
    """
    Splits the training data into accumulated datasets and trains the model on each subset.

    Parameters:
    - X_train: Training input features (numpy array)
    - y_train: Training targets (numpy array)
    - X_val: Validation input features (numpy array)
    - y_val: Validation targets (numpy array)
    - number_of_accumulated_datasets: Number of accumulated datasets
    - model_arch: C.NET_ARCH_SIMPLE_FULLY_CONNECTED or 'C.NET_ARCH_LSTM'
    - device: 'cpu' or 'cuda'

    Returns:
    - sample_sizes: List of sample sizes used for training
    - accuracies: List of validation MAEs corresponding to each sample size
    """
    total_samples = X_train.shape[0]
    increment = total_samples // number_of_accumulated_datasets
    sample_sizes = []
    
    accuracies_val = {}
    accuracies_train = {}
    
    for i in range(1, number_of_accumulated_datasets + 1):
        current_size = increment * i
        if i == number_of_accumulated_datasets:
            current_size = total_samples  # Ensure all samples are used in the last dataset
        X_subset = X_train[:current_size]
        y_subset = y_train[:current_size]
        
        print(f'\nTraining on dataset {i}/{number_of_accumulated_datasets} with {current_size} samples')

        # Create TensorDataset and DataLoader
        subset_dataset = TensorDataset(torch.tensor(X_subset, dtype=torch.float32),
                                      torch.tensor(y_subset, dtype=torch.float32))
        subset_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)
        
        # Create validation DataLoader
        val_subset_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                          torch.tensor(y_val, dtype=torch.float32))
        val_subset_loader = DataLoader(val_subset_dataset, batch_size=batch_size, shuffle=False)
        
        # Instantiate a new model
        model_acc = build_model(model_arch, data_shapes, device)
        
        # Train the model
        training_results = train_model(
            model=model_acc,
            train_loader=subset_loader,
            val_loader=val_subset_loader,
            device=device,
            data_sub_cfg=data_sub_cfg,
            num_epochs=num_epochs,
            folder_prefix=f'accumulated_datasets_{i}'
        )
        
        train_losses_acc = training_results['train_losses']
        val_losses_acc = training_results['val_losses']
        train_maes_acc = training_results['train_maes']
        val_maes_acc = training_results['val_maes']
        
        val_avg_mae_acc = np.array(val_maes_acc).mean()
        train_avg_maes_acc = np.array(train_maes_acc).mean()
        
        print(f'Dataset {i} Validation MAE: {val_avg_mae_acc:.6f}')
        
        # Record the results
        sample_sizes.append(current_size)
        accuracies_val[current_size] = val_avg_mae_acc
        accuracies_train[current_size] = train_avg_maes_acc
    
    return sample_sizes, accuracies_val, accuracies_train


# Run the accumulated datasets training benchmark
if __name__ == '__main__':
    
    # model_arch = C.NET_ARCH_LSTM
    
    model_arch = C.NET_ARCH_LSTM
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_batch_size = 16
    num_epochs = 600
    number_of_accumulated_datasets = 10
    
    # data_dir = 'data/Sim3000_2000turns_10parts_FODOErr-123457-126-017_avgTrue_tgtquad_misalign_deltas_2'
    
    # data_dir = 'data/Sim1000_2000turns_10parts_FODOErr-123457--_avgTrue_tgtquad_misalign_deltas_1'
    
    data_dir = 'data/Sim2000_6000turns_10parts_FODOErr-123457-136-_avgTrue_tgtquad_misalign_deltas_1'
    
    sim_data = load_data_from_dir(data_dir)
    data_sub_cfg = get_data_sub_cfg(sim_data)
    

    input_tensors_scaled = sim_data[C.DATA_KEY_INPUT_TENSORS_SCALED]
    target_tensors_scaled = sim_data[C.DATA_KEY_TARGET_TENSORS_SCALED]
    dataset_scalers = sim_data[C.DATA_KEY_DATASET_SCALERS]
    
    train_inputs, val_inputs, train_targets, val_targets, data_shapes = get_data_splits(sim_data, test_size=0.10, model_arch=model_arch)

    # Run accumulated datasets training
    sample_sizes, accuracies_val, accuracies_train = train_accumulated_datasets(
        X_train=train_inputs,
        y_train=train_targets,
        X_val=val_inputs,
        y_val=val_targets,
        batch_size=train_batch_size,
        data_shapes=data_shapes,
        data_sub_cfg=data_sub_cfg,
        number_of_accumulated_datasets=number_of_accumulated_datasets,
        model_arch=model_arch,
        device=device
    )

    benchmark_results = {
        C.KEY_ACCUMULATED_DATASETS: {
            "nb_datasets": number_of_accumulated_datasets,
            "results_val_mae": accuracies_val,
            "results_train_mae": accuracies_train
        }
    }

    benchmark_results[C.KEY_ACCUMULATED_DATASETS]['results_val_mae_unscaled'] = {}
    benchmark_results[C.KEY_ACCUMULATED_DATASETS]['results_train_mae_unscaled'] = {}

    mean_min, mean_max = dataset_scalers['target_scaler'].data_min_.mean(), dataset_scalers['target_scaler'].data_max_.mean()
    mean_scaler = mean_max - mean_min

    for ii, accc in accuracies_val.items():
        mean_unscaled = accc * mean_scaler
        print(f"{mean_unscaled:.7f}", f"{mean_unscaled * 1e6:.2f}")
        benchmark_results[C.KEY_ACCUMULATED_DATASETS]['results_val_mae_unscaled'][ii] = mean_unscaled

    for ii, accc in accuracies_train.items():
        mean_unscaled = accc * mean_scaler
        print(f"{mean_unscaled:.7f}", f"{mean_unscaled * 1e6:.2f}")
        benchmark_results[C.KEY_ACCUMULATED_DATASETS]['results_train_mae_unscaled'][ii] = mean_unscaled


    torch.save(benchmark_results, f"{SAVE_DIR_BENCHMARKS}/benchmark_results_accumulated_datasets.pt")
    
    plot_benchmark_accumulated_datasets(benchmark_results)


