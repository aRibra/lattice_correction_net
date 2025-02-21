# visualization.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from collections import defaultdict
from pylab import *

from constants import Constants as C
from sim_config import SAVE_DIR_FIGS, SAVE_DIR_BENCHMARKS


def print_maes_micron(val_maes, target_scaler):
    mean_min, mean_max = target_scaler.data_min_.mean(), target_scaler.data_max_.mean()
    mean_scaler = mean_max - mean_min
    
    val_maes_torch = torch.from_numpy(np.array(val_maes))
    val_maes_microns = val_maes_torch * mean_scaler * 1e6
    
    print("val_maes_microns: ", val_maes_microns)


import matplotlib.pyplot as plt
import numpy as np

# Assuming C and SAVE_DIR_FIGS are defined elsewhere in your code
# For example:
# class C:
#     DATA_KEY_ALL_ERROR_VALUES_DIPOLE_TILT = 'dipole_tilt_errors'
#     DATA_KEY_ALL_ERROR_VALUES_QUAD_TILT = 'quad_tilt_errors'
#     DATA_KEY_DATA_AUTOMATION = 'data_automation'
#     DATA_KEY_TARGET_TENSORS = 'target_tensors'
#     DATA_KEY_MERGED_CONFIG = 'merged_config'
#
# SAVE_DIR_FIGS = '/path/to/save/figures'
import matplotlib.pyplot as plt
import numpy as np

def plot_data_histograms(sim_data,
                         plot_com_deltas_x=True,
                         plot_com_deltas_y=True,
                         plot_dipole_tilt_error_hist=True,
                         plot_quad_tilt_error_hist=True):
    
    all_error_values_dipole_tilt = sim_data[C.DATA_KEY_ALL_ERROR_VALUES_DIPOLE_TILT]
    all_error_values_quad_tilt = sim_data[C.DATA_KEY_ALL_ERROR_VALUES_QUAD_TILT]
    data_automation = sim_data[C.DATA_KEY_DATA_AUTOMATION]
    target_tensors = sim_data[C.DATA_KEY_TARGET_TENSORS]

    merged_config = sim_data[C.DATA_KEY_MERGED_CONFIG]

    print("merged_config: ", merged_config)

    if merged_config['quad_errors']:
        fodo_indices_with_error = [err['FODO_index'] for err in merged_config['quad_errors']]
    else:
        fodo_indices_with_error = []
        
    if merged_config['quad_tilt_errors']:
        fodo_indices_with_quad_tilt_error = [err['FODO_index'] for err in merged_config['quad_tilt_errors']]
    else:
        fodo_indices_with_quad_tilt_error = []
        
    if merged_config['dipole_tilt_errors']:        
        fodo_indices_with_dipole_tilt_error = [err['FODO_index'] for err in merged_config['dipole_tilt_errors']]
    else:
        fodo_indices_with_dipole_tilt_error = []

    # --- Add Conversion Functions ---
    def mrad_to_deg(x):
        """Convert milliradians to degrees."""
        return x * (180 / np.pi) / 1000  # mrad to degrees

    def deg_to_mrad(x):
        """Convert degrees to milliradians."""
        return x * (np.pi / 180) * 1000  # degrees to mrad

    # --- Single Histogram Plot ---
    plt.figure(figsize=(7, 6))
    plt.hist(target_tensors * 1e6, bins=8, edgecolor='black')

    plt.title('delta_y error hist', fontsize=18)
    plt.xlabel('micron', fontsize=18)
    plt.ylabel('n_samples/bin', fontsize=18)
    plt.tick_params(axis='both', labelsize=14)  # Set font size for ticks
    # Removed the second tick_params call to avoid redundancy
    legend_quad_offset = [f"quad_{qix}" for qix in fodo_indices_with_error]
    if legend_quad_offset:  # Check if the legend list is not empty
        plt.legend(legend_quad_offset, fontsize=15)
    plt.minorticks_on()
    plt.grid()
    plt.savefig(f"{SAVE_DIR_FIGS}/delta_y_error_hist.eps", bbox_inches='tight', format='eps')
    plt.show()

    # --- Individual Histograms ---
    if plot_com_deltas_x:
        plt.figure(figsize=(7, 6))
        plt.hist(np.array(data_automation.com_deltas_x) * 1e6, bins=30, edgecolor='black', color='skyblue')
        plt.title('COM Deltas X Histogram', fontsize=18)
        plt.xlabel('Micron', fontsize=16)
        plt.ylabel('Number of Samples per Bin', fontsize=16)
        plt.legend(['x'], fontsize=14)
        plt.tick_params(axis='both', labelsize=14)
        plt.minorticks_on()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR_FIGS}/com_deltas_x_hist.eps", bbox_inches='tight', format='eps')
        plt.show()

    if plot_com_deltas_y:
        plt.figure(figsize=(7, 6))
        plt.hist(np.array(data_automation.com_deltas_y) * 1e6, bins=30, edgecolor='black', color='salmon')
        plt.title('COM Deltas Y Histogram', fontsize=18)
        plt.xlabel('Micron', fontsize=16)
        plt.ylabel('Number of Samples per Bin', fontsize=16)
        plt.legend(['y'], fontsize=14)
        plt.tick_params(axis='both', labelsize=15)
        plt.minorticks_on()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR_FIGS}/com_deltas_y_hist.eps", bbox_inches='tight', format='eps')
        plt.show()

    if plot_dipole_tilt_error_hist:
        plt.figure(figsize=(7, 6))
        plt.hist(all_error_values_dipole_tilt * 1e3, bins=20, edgecolor='black')
        plt.title('Dipoles Tilt Error Histogram', fontsize=18)
        plt.xlabel('Tilt Angle (mrad)', fontsize=16)
        plt.ylabel('Number of Samples per Bin', fontsize=18)
        legend_dipole_tilt = [f"dipole_{qix}" for qix in fodo_indices_with_dipole_tilt_error]
        if legend_dipole_tilt:  # Check if the legend list is not empty
            plt.legend(legend_dipole_tilt, fontsize=15)
        plt.tick_params(axis='both', labelsize=15)
        plt.minorticks_on()
        plt.grid()
        # Add Secondary X-Axis for Dipoles Tilt Error Histogram
        secax_dipole = plt.gca().secondary_xaxis('top', functions=(mrad_to_deg, deg_to_mrad))
        secax_dipole.set_xlabel('Tilt Angle (degrees)', fontsize=16)
        secax_dipole.tick_params(axis='x', labelsize=15)
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR_FIGS}/dipole_tilt_error_hist.eps", bbox_inches='tight', format='eps')
        plt.show()

    if plot_quad_tilt_error_hist:
        plt.figure(figsize=(7, 6))
        plt.hist(all_error_values_quad_tilt * 1e3, bins=20, edgecolor='black')
        plt.title('Quadrupoles Tilt Error Histogram', fontsize=18)
        plt.xlabel('Tilt Angle (mrad)', fontsize=16)
        plt.ylabel('Number of Samples per Bin', fontsize=18)
        legend_quad_tilt = [f"quad_{qix}" for qix in fodo_indices_with_quad_tilt_error]
        if legend_quad_tilt:  # Check if the legend list is not empty
            plt.legend(legend_quad_tilt, fontsize=15)
        plt.tick_params(axis='both', labelsize=15)
        plt.minorticks_on()
        plt.grid()
        # Add Secondary X-Axis for Quadrupoles Tilt Error Histogram
        secax_quad = plt.gca().secondary_xaxis('top', functions=(mrad_to_deg, deg_to_mrad))
        secax_quad.set_xlabel('Tilt Angle (degrees)', fontsize=16)
        secax_quad.tick_params(axis='x', labelsize=15)
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR_FIGS}/quad_tilt_error_hist.eps", bbox_inches='tight', format='eps')
        plt.show()

    
def get_local_plt_markers(nb_binz):
    plt_markers = Line2D.markers.copy()
    plt_markers.pop('')
    plt_markers.pop(' ')
    plt_markers.pop('None')
    plt_markers.pop('none')
    local_plt_markers_list = [pm for pm in plt_markers.keys()]
    
    if nb_binz > len(plt_markers):
        nb_repeat = int(ceil(nb_binz / len(plt_markers)))
        local_plt_markers_list = local_plt_markers_list * nb_repeat
        local_plt_markers_list = local_plt_markers_list[:nb_binz]
    
    return local_plt_markers_list


def plot_benchmark_stats(stats, benchmark_info):
    noise_range = (benchmark_info["noise_start"], benchmark_info["noise_stop"])
    bins = benchmark_info["bins"]
    noise_pallette = benchmark_info["noise_pallette"]
    runs_per_noise = benchmark_info["runs_per_noise"]
    fodo_mapping = benchmark_info["fodo_mapping"]
    cancel_tilt_error = benchmark_info["cancel_tilt_error"]
    cancel_misalign_error = benchmark_info["cancel_misalign_error"]

    mean_errors = defaultdict(dict)
    std_errors = defaultdict(dict)
    
    for noise_level in stats:
        try:
            float(noise_level)
        except:
            continue
        for fodo_ix in stats[noise_level]:
            mean_errors[noise_level][fodo_ix] = np.mean(stats[noise_level][fodo_ix])
            std_errors[noise_level][fodo_ix] = np.std(stats[noise_level][fodo_ix])

    # Plotting accuracy vs noise_level for each FODO index
    plt.figure(figsize=(12, 8))
    fodo_indices = set()
    for noise_level in stats:
        fodo_indices.update(stats[noise_level].keys())
    fodo_indices = sorted(list(fodo_indices)[: len(benchmark_info['fodo_mapping']) ])

    local_plt_markers = get_local_plt_markers(len(fodo_indices))
    for fodo_ix in fodo_indices:
        means = [mean_errors[noise][fodo_ix] for noise in noise_pallette]
        means = np.array(means)
        marker = local_plt_markers[fodo_ix]
        plt.plot(noise_pallette * 1e3, means * 1e3, '-' + marker, label=f'FODO-QE-ix {fodo_mapping[fodo_ix]}')

    
    plt.xlabel('Noise Level (µrad)', fontsize=18)
    plt.ylabel(f'Mean Absolute Error (µrad)\nAveraging over {runs_per_noise} simulation runs\n', fontsize=18)
    plt.title(f'Model Prediction Accuracy vs BPM Noise Level (noise_range: {noise_range}, bins: {bins})\nMisAlign error: {not cancel_misalign_error}\nTilt errors: {not cancel_tilt_error}', fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True)
    # plt.minorticks_on()
    plt.xticks(noise_pallette * 1e3, fontsize=14, rotation=-45)
    plt.yticks(fontsize=14)
    plt.rc('font', size=14)

    
    # Create inset axes
    ax_main = plt.gca()
    ax_inset = inset_axes(ax_main, width="30%", height="10%", loc='lower right', borderpad=2.5)

    # Extract MAE for noise_level=0.0
    noise_zero_key = 0.0  # Adjust if noise levels are stored as floats
    if noise_zero_key in mean_errors:
        mae_zero = [mean_errors[noise_zero_key].get(fodo_ix, 0) * 1e6 for fodo_ix in fodo_indices]
        fodo_labels = [f'{fodo_mapping[fodo_ix]}' for fodo_ix in fodo_indices]
        fodo_labels_int = [fodo_mapping[fodo_ix] for fodo_ix in fodo_indices]

        # Plotting the inset bar chart
        ax_inset.bar(fodo_labels, mae_zero, color='skyblue', edgecolor='black')

        bars = ax_inset.bar(fodo_labels, mae_zero, color='skyblue', edgecolor='black')

        # Add text labels on each bar
        for bar in bars:
            height = bar.get_height()
            ax_inset.text(
                bar.get_x() + bar.get_width() / 2.,  # X-coordinate: center of the bar
                height - 0.015,                              # Y-coordinate: top of the bar
                f'{height:.2f}',                     # Text: MAE value with 3 decimals
                ha='center',                         # Horizontal alignment
                va='baseline',                         # Vertical alignment
                fontsize=11,
                color='black'
            )
        
        # ax_inset.plot(fodo_labels_int, mae_zero, 's')
        # ax_inset.plot(np.array(fodo_labels_int), np.array(mae_zero), '-')
        ax_inset.set_title('MAE at Noise=0', fontsize=11)
        ax_inset.set_ylabel('MAE (µrad)', fontsize=11)
        ax_inset.set_xlabel('FODO Index', fontsize=11)
        ax_inset.tick_params(axis='both', which='major', labelsize=11)
        # plt.yticks(mae_zero)
        # plt.xticks(fodo_labels_int)
        ax_inset.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        
    plt.savefig(f"{SAVE_DIR_FIGS}/plot_accuracy_benchmark_with_MisAlign_and_Tilt_error_in_data_1.eps", bbox_inches = 'tight', format='eps')
    plt.show()
    

def plot_training_results(num_epochs, train_losses, val_losses, train_maes, val_maes):
    """
    Plot the training and validation losses and Mean Absolute Errors (MAE) for the given model.
    Args:
        num_epochs (int): Number of training epochs.
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
        train_maes (list): List of training MAEs.
        val_maes (list): List of validation MAEs.
    Returns:
        None
    """    
    
    # Plot training and validation loss
    plt.figure(figsize=(10,5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=18 )
    plt.ylabel('Loss', fontsize=18)
    plt.title('Training and Validation Loss', fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.savefig(f"{SAVE_DIR_FIGS}/train_loss.eps", bbox_inches = 'tight', format='eps')
    plt.show()
    
    # Plot training and validation MAE
    plt.figure(figsize=(10,5))
    plt.plot(range(1, num_epochs+1), train_maes, label='Train MAE')
    plt.plot(range(1, num_epochs+1), val_maes, label='Validation MAE')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Mean Absolute Error', fontsize=18)
    plt.title('Training and Validation MAE', fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.savefig(f"{SAVE_DIR_FIGS}/acc.eps", bbox_inches = 'tight', format='eps')
    plt.show()
    
    
    # Plot training and validation loss
    plt.figure(figsize=(10,5))
    loglog(train_losses, label='Train Loss')
    loglog(val_losses, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.title('Training and Validation Loss', fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.savefig(f"{SAVE_DIR_FIGS}/train_loss_loglog.eps", bbox_inches = 'tight', format='eps')
    plt.show()
    
    # Plot training and validation MAE
    plt.figure(figsize=(10,5))
    loglog(train_maes, label='Train MAE')
    loglog(val_maes, label='Validation MAE')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Mean Absolute Error', fontsize=18)
    plt.title('Training and Validation MAE', fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.savefig(f"{SAVE_DIR_FIGS}/acc_loglog.eps", bbox_inches = 'tight', format='eps')
    plt.show()

# Function to scale tensor from [-1,1] to [0,1]
def scale_tensor(tensor):
    return (tensor + 1) / 2

# Function to prepare RGB image by combining two channels
def prepare_rgb_image(tensor1, tensor2):
    tensor1 = scale_tensor(tensor1)
    tensor2 = scale_tensor(tensor2)
    
    # Initialize RGB image with zeros
    rgb_image = np.zeros((tensor1.shape[0], tensor1.shape[1], 3))
    
    # Assign channels
    rgb_image[:, :, 0] = tensor1  # Red
    rgb_image[:, :, 1] = tensor2  # Green
    # Blue channel remains zero
    
    return rgb_image

# Function to visualize a single sample's channels
def visualize_sample_channels(sample, sample_idx):
    # Separate channels
    channel_1 = scale_tensor(sample[0]).cpu().numpy()
    channel_2 = scale_tensor(sample[1]).cpu().numpy()
    
    # Plot channels side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(channel_1, cmap='gray', aspect='auto')
    axes[0].set_title(f'Sample {sample_idx} - Channel 1')
    axes[0].axis('off')
    
    axes[1].imshow(channel_2, cmap='gray', aspect='auto')
    axes[1].set_title(f'Sample {sample_idx} - Channel 2')
    axes[1].axis('off')
    
    plt.show()

# Function to visualize a single sample as RGB
def visualize_sample_rgb(sample, sample_idx):
    # Prepare RGB image
    rgb_image = prepare_rgb_image(sample[0], sample[1])
    
    # Plot RGB image
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb_image, aspect='auto')
    plt.title(f'Sample {sample_idx} - Combined Channels (RGB)')
    plt.axis('off')
    plt.show()

# Function to visualize a grid of samples' channels
def visualize_grid_channels(samples, start_idx=1):
    num_samples = samples.shape[0]
    num_cols = 2  # Two channels per sample
    num_rows = num_samples
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 5 * num_rows))
    
    for i in range(num_samples):
        sample = samples[i]
        channel_1 = scale_tensor(sample[0]).cpu().numpy()
        channel_2 = scale_tensor(sample[1]).cpu().numpy()
        
        # Plot Channel 1
        axes[i, 0].imshow(channel_1, cmap='gray', aspect='auto')
        axes[i, 0].set_title(f'Sample {start_idx + i} - Channel 1')
        axes[i, 0].axis('off')
        
        # Plot Channel 2
        axes[i, 1].imshow(channel_2, cmap='gray', aspect='auto')
        axes[i, 1].set_title(f'Sample {start_idx + i} - Channel 2')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Function to visualize a grid of samples as RGB
def visualize_grid_rgb(samples, start_idx=1):
    """
    Example Usage:

    # Select samples to visualize
    start_ix_viz = np.random.randint(input_data_cnn.shape[0])
    num_samples_to_visualize = 1
    samples = input_data_cnn[start_ix_viz:start_ix_viz+num_samples_to_visualize]

        # for i in range(num_samples_to_visualize):
        #     sample = samples[i]
        #     visualize_sample_channels(sample, i+1)
        #     visualize_sample_rgb(sample, i+1)
    """
    num_samples = samples.shape[0]
    fig, axes = plt.subplots(1, num_samples, figsize=(6 * num_samples, 6))
    
    for i in range(num_samples):
        sample = samples[i]
        rgb_image = prepare_rgb_image(sample[0], sample[1])
        
        axes[i].imshow(rgb_image, aspect='auto')
        axes[i].set_title(f'Sample {start_idx + i} - RGB')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_benchmark_accumulated_datasets(benchmark_results):
    """ Plot the results of the accumulated datasets benchmark
    Results are saved in the benchmark_results dictionary
    """
    
    all_dataset_chunks_sizes = list(benchmark_results['accumulated_datasets']['results_val_mae'].keys())
    chunk_indices = np.arange(len(all_dataset_chunks_sizes)) + 1

    # Plot Validation MAE Across Folds
    print("\nGenerating Validation MAE Across Folds Plot\n")
    plt.figure(figsize=(8,6))
    plt.bar(chunk_indices, all_dataset_chunks_sizes, color='skyblue')
    plt.xlabel('Chunk index', fontsize=14)
    plt.ylabel('Chunk number of samples', fontsize=14)
    plt.title('Number of Samples Across Dataset Chunks', fontsize=18)
    plt.xticks(chunk_indices, fontsize=14)
    plt.yticks(all_dataset_chunks_sizes, fontsize=14)
    plt.grid(axis='y')
    plt.savefig(f"{SAVE_DIR_BENCHMARKS}/plot_benchmark_accumulated_training_chunk_sizes.eps", bbox_inches = 'tight', format='eps')
    plt.show()


    # Plot Accuracy vs Number of Samples
    plt.figure(figsize=(10,6))
    markers = ['d', 'o']
    for ix, acc_result in enumerate([
            benchmark_results['accumulated_datasets']['results_train_mae_unscaled'], 
            benchmark_results['accumulated_datasets']['results_val_mae_unscaled'] ]):
        sample_sizes = acc_result.keys()
        accuracies = list(acc_result.values())
        accuracies = np.array(accuracies) * 1e6
        marker = markers[ix]
        plt.plot(sample_sizes, accuracies, marker=marker, linestyle='-')
        plt.xticks(list(sample_sizes), fontsize=14)
        plt.yticks(fontsize=14)
    # plt.ylim([3, 33])
    plt.xlabel('Number of Training Samples', fontsize=18)
    plt.ylabel('MAE (µm)', fontsize=18)
    plt.title('Train/Val MAE vs Number of Training Samples', fontsize=18)
    plt.legend(['train MAE', 'val MAE'], fontsize=14)
    plt.grid(True)
    plt.minorticks_off()
    plt.savefig(f"{SAVE_DIR_BENCHMARKS}/plot_bechmark_accumuluated_datasets_Mixed-data.eps", bbox_inches = 'tight', format='eps')
    plt.show()



def plot_accuracy_per_fold(benchmark_results):
    """
    Plots the accuracy per fold for the cross-validation results.
    """
    # Extract fold numbers and their corresponding accuracies
    fold_numbers = list(benchmark_results['cross_validation']['results_val_mae_unscaled'].keys())
    accuracies = list(benchmark_results['cross_validation']['results_val_mae_unscaled'].values())
    accuracies = np.array(accuracies) * 1e6

    # Plot Accuracy per Fold
    plt.figure(figsize=(10, 6))
    plt.plot(fold_numbers, accuracies, marker='o', linestyle='-', color='b', label='MAE per Fold')

    # Customize the plot
    plt.xticks(fold_numbers, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Fold Number', fontsize=18)
    plt.ylabel('MAE (µm)', fontsize=18)
    plt.title('Cross-Validation MAE per Fold', fontsize=20)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.minorticks_off()
    # plt.ylim([4.5, 8])

    plt.savefig(f"{SAVE_DIR_BENCHMARKS}/cross_validation_accuracy_folds.eps", bbox_inches='tight', format='eps')

    plt.show()


def plot_cv_indices(cv, X, y, train_inputs_shape, n_splits, lw=10):
    
    fig, ax = plt.subplots()

    print(train_inputs_shape)

    nb_samples = train_inputs_shape[0]
    cmap_data = plt.cm.Paired
    cmap_cv = plt.cm.coolwarm
    """Create a sample plot for indices of a cross-validation object."""
    
    # Generate the training/testing visualizations for each CV split
    splits = list(cv.split(X=X, y=y))
    splits_indices = np.arange(len(splits))[::-1]
    
    for ii in splits_indices:
        (tr, tt) = splits[ii]
        # Fill in indices with the training/test 
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [len(splits) - ii - 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    # # Formatting
    yticklabels = list(range(n_splits))
    ax.set(
        yticks=np.arange(n_splits) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        xlim=[0, nb_samples]
    )
    ax.set_title("{} Cross-validation".format(type(cv).__name__), fontsize=18)
    # plt.rc('font', size=10)
    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(18)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    ax.legend(
        [Patch(color=cmap_cv(0.8)), Patch(color=cmap_cv(0.02))],
        ["Testing set", "Training set"],
        loc=(1.02, 0.8),
    )

    plt.savefig(f"{SAVE_DIR_BENCHMARKS}/plot_cross_validation_train_test_folds.eps", bbox_inches = 'tight', format='eps')
    plt.show()
