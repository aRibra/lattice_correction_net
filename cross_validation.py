# cross_validation.py

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import matplotlib

from sim_config import SAVE_DIR_BENCHMARKS, SHOW_PLOTS
from net import build_model, train_model
from data import load_data_from_dir, get_data_splits, get_data_sub_cfg

from visualization import plot_accuracy_per_fold, plot_cv_indices
from constants import Constants as C

if not SHOW_PLOTS:
    matplotlib.use('Agg')


def perform_cross_validation(X_train,
                             y_train,
                             data_shapes,
                             data_sub_cfg,
                             model_arch=C.NET_ARCH_LSTM,
                             device='cuda',
                             k_folds=5,
                             batch_size=16,
                             num_epochs=600):
    """
    Performs K-Fold Cross-Validation to evaluate model performance.

    Parameters:
    - X_train: Training input features (numpy array)
    - y_train: Training targets (numpy array)
    - data_shapes: Shapes of the data (from get_data_splits)
    - data_sub_cfg: Data sub-configuration (from get_data_sub_cfg)
    - model_arch: Model architecture (default: C.NET_ARCH_LSTM)
    - device: 'cpu' or 'cuda' (default: 'cuda')
    - k_folds: Number of cross-validation folds (default: 5)
    - batch_size: Batch size for training (default: 16)
    - num_epochs: Number of training epochs per fold (default: 600)

    Returns:
    - cv_results: Dictionary containing validation MAEs per fold
    """
    kf = KFold(n_splits=k_folds, shuffle=False)
    cv_results = {}
    fold = 1

    # Optionally, plot the CV splits
    plot_cv_indices(kf, X_train, y_train, data_shapes['inputs_shape'], n_splits=k_folds, lw=20)

    for train_index, val_index in kf.split(X_train):
        current_fold = fold
        print(f'\nStarting Fold {current_fold}/{k_folds}')

        # Split data
        X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
        y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]

        # Create TensorDatasets and DataLoaders
        train_dataset = TensorDataset(torch.tensor(X_train_cv, dtype=torch.float32),
                                      torch.tensor(y_train_cv, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val_cv, dtype=torch.float32),
                                    torch.tensor(y_val_cv, dtype=torch.float32))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        print(f'Training on Fold {current_fold}: {X_train_cv.shape[0]} training samples, {X_val_cv.shape[0]} validation samples')

        # Instantiate a new model for each fold
        model = build_model(model_arch, data_shapes, device)

        # Train the model
        training_results = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            data_sub_cfg=data_sub_cfg,
            num_epochs=num_epochs,
            folder_prefix=f'fold_{current_fold}'
        )

        val_maes = training_results['val_maes']
        avg_val_mae = np.mean(val_maes)

        print(f'Fold {current_fold} Validation MAE: {avg_val_mae:.6f}')

        # Record the results
        cv_results[current_fold] = avg_val_mae
        fold += 1

    return cv_results


# Run the cross-validation benchmark
if __name__ == '__main__':

    model_arch = C.NET_ARCH_LSTM  # You can change this to C.NET_ARCH_SIMPLE_FULLY_CONNECTED if needed

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16
    num_epochs = 600
    k_folds = 5

    # data_dir = 'data/Sim1000_2000turns_10parts_FODOErr-123457--_avgTrue_tgtquad_misalign_deltas_1'
    data_dir = 'data/Sim2000_6000turns_10parts_FODOErr-123457-136-_avgTrue_tgtquad_misalign_deltas_1'

    # Load and prepare data
    sim_data = load_data_from_dir(data_dir)
    data_sub_cfg = get_data_sub_cfg(sim_data)

    X = sim_data[C.DATA_KEY_INPUT_TENSORS_SCALED]
    y = sim_data[C.DATA_KEY_TARGET_TENSORS_SCALED]
    dataset_scalers = sim_data[C.DATA_KEY_DATASET_SCALERS]

    train_inputs, val_inputs, train_targets, val_targets, data_shapes = get_data_splits(sim_data, test_size=0.0001, model_arch=model_arch)

    # Perform cross-validation
    cv_results = perform_cross_validation(
        X_train=train_inputs,
        y_train=train_targets,
        data_shapes=data_shapes,
        data_sub_cfg=data_sub_cfg,
        model_arch=model_arch,
        device=device,
        k_folds=k_folds,
        batch_size=batch_size,
        num_epochs=num_epochs
    )

    # Prepare benchmark results
    benchmark_results = {
        "cross_validation": {
            "k_folds": k_folds,
            "results_val_mae": cv_results
        }
    }

    benchmark_results['cross_validation']['results_val_mae_unscaled'] = {}

    mean_min = dataset_scalers['target_scaler'].data_min_.mean()
    mean_max = dataset_scalers['target_scaler'].data_max_.mean()
    mean_scaler = mean_max - mean_min

    for fold, mae in cv_results.items():
        mean_unscaled = mae * mean_scaler
        print(f"Fold {fold} Unscaled Validation MAE: {mean_unscaled:.7f}, {mean_unscaled * 1e6:.2f}")
        benchmark_results['cross_validation']['results_val_mae_unscaled'][fold] = mean_unscaled

    # Save benchmark results
    torch.save(benchmark_results, f"{SAVE_DIR_BENCHMARKS}/benchmark_results_cross_validation.pt")

    # Plot cross-validation results
    plot_accuracy_per_fold(benchmark_results)

