# eval.py

import copy
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import defaultdict
from constants import Constants as C
from visualization import plot_benchmark_stats
from utils import convert_defaultdict_to_dict, deserialize_minmax_scaler

from synchrotron_simulator_gpu_Dataset_4D import SimulationRunner
from automate_dataset_collection import SimulationDataset
from sim_config import SAVE_DIR_BENCHMARKS, SAVE_DIR_FIGS



def inference_on_validation_data(model, val_loader, dataset_scalers, merged_config):
    """
    Perform inference on the validation data using the trained model.
    Args:
        model (torch.nn.Module): The trained model for inference.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        dataset_scalers (dict): Dictionary containing the MinMaxScaler objects for inputs and targets.
        merged_config (dict): base simulation configurations with quadrupole errors.
    Returns:
        None
    """
    
    all_mean_min, all_mean_max = dataset_scalers['target_scaler'].data_min_, dataset_scalers['target_scaler'].data_max_
    all_mean_scaler = all_mean_max - all_mean_min


    # Extract FODO cell indices
    if merged_config['target_data'] == 'quad_misalign_deltas':
        fodo_cell_indices = [err['FODO_index'] for err in merged_config['quad_errors']]
    elif merged_config['target_data'] == 'quad_tilt_angles':
        fodo_cell_indices = [err['FODO_index'] for err in merged_config['quad_tilt_errors']]
    elif merged_config['target_data'] == 'dipole_tilt_angles':
        fodo_cell_indices = [err['FODO_index'] for err in merged_config['dipole_tilt_errors']]
    
    print(fodo_cell_indices)

    # Batch parameters
    batch_limit_s = 0
    batch_limit_e = 16
    nb_batches = 4
    batch_counter = 0

    # Number of columns for subplots
    cols = 3
    rows = math.ceil(len(fodo_cell_indices) / cols)  # Calculate the number of rows required

    for batch_inputs, batch_targets in val_loader:  # train_loader

        if batch_counter == nb_batches:
            break

        batch_counter += 1

        print(batch_inputs.shape)
        with torch.no_grad():
            # Forward pass
            output = model(batch_inputs[batch_limit_s:batch_limit_e].cuda())

        output = output.cpu()

        # Scale the targets and outputs
        batch_targets_scaled = batch_targets * all_mean_scaler * 1e6
        output_scaled = output * all_mean_scaler * 1e6

        # Calculate residuals
        err_resid = output_scaled - batch_targets_scaled[batch_limit_s:batch_limit_e]

        # Create the main figure
        fig = plt.figure(figsize=(20, 6 * rows))  # Increased height for larger plots
        # Create a top-level GridSpec with reduced spacing
        main_gs = GridSpec(rows, cols, figure=fig, wspace=0.3, hspace=0.3)

        # Supertitle
        fig.suptitle(
            "Prediction Samples Vs Ground Truth on Validation Data",
            fontsize=20,          # Adjust font size as needed
            y=0.95,                # Adjust y-position to prevent overlap
            fontweight='bold'      # Optional: Make the title bold
        )
        
        for idx, quad_idx_pred in enumerate(range(len(fodo_cell_indices))):
            # Determine the row and column for the current subplot
            row = idx // cols
            col = idx % cols

            # Access the specific GridSpec cell
            cell_gs = main_gs[row, col]
            # Create a nested GridSpec within the cell (2 rows: main and residual)
            nested_gs = cell_gs.subgridspec(2, 1, height_ratios=[4, 1], hspace=0.05)

            # Create the main plot axes
            main_ax = fig.add_subplot(nested_gs[0])
            # Create the residual plot axes
            resid_ax = fig.add_subplot(nested_gs[1], sharex=main_ax)

            # Plotting on the main axes
            main_ax.plot(
                batch_targets_scaled.cpu()[:, quad_idx_pred],
                '-gs',
                lw=5,  # Increased line width for better visibility
                alpha=0.5,
                label='Ground Truth'
            )
            main_ax.plot(
                output_scaled[:, quad_idx_pred].cpu(),
                '-.b',
                lw=2,
                label='Prediction'
            )
            main_ax.plot(
                err_resid.cpu()[:, quad_idx_pred],
                '-or',
                lw=1,
                label='Residual Error'
            )
            main_ax.legend(['gt', 'pred', 'err_resid'], fontsize=14)
            main_ax.set_title(
                f"FODO Cell Index: {fodo_cell_indices[quad_idx_pred]}",
                fontsize=18
            )
            main_ax.set_ylabel("Predicted Error\n(µm)", fontsize=18)
            main_ax.tick_params(axis='both', labelsize=15)  # Set font size for ticks
            main_ax.minorticks_on()

            # Plotting on the residual axes
            resid_ax.plot(
                err_resid.cpu()[:, quad_idx_pred],
                '-or',
                lw=1,
                label='Residual Error'
            )
            resid_ax.legend(['Residual'], fontsize=14)
            resid_ax.set_ylabel("Residual\n(µm)", fontsize=18)
            resid_ax.set_xlabel("Batch Sample", fontsize=15)
            resid_ax.tick_params(axis='both', labelsize=15)  # Set font size for ticks
            resid_ax.minorticks_on()

            # Optional: Adjust y-limits for residuals to focus on their scale
            # resid_min = err_resid.cpu()[:, quad_idx_pred].min()
            # resid_max = err_resid.cpu()[:, quad_idx_pred].max()
            # resid_ax.set_ylim(resid_min * 1.1, resid_max * 1.1)

        # Hide any unused subplots if the grid has more cells than FODO indices
        total_subplots = rows * cols
        if len(fodo_cell_indices) < total_subplots:
            for idx in range(len(fodo_cell_indices), total_subplots):
                row = idx // cols
                col = idx % cols
                cell_gs = main_gs[row, col]
                # Create an empty axes and hide it
                empty_ax = fig.add_subplot(cell_gs)
                empty_ax.axis('off')

        plt.tight_layout()
        plt.savefig(
            f"{SAVE_DIR_FIGS}/inference_val_batch_{batch_counter}.eps",
            bbox_inches='tight',
            format='eps'
        )
        plt.show()


def _run_evaluation(model, base_configurations, common_parameters, dataset_scalers, noise_type=None, noise_level=0.0, plot=False, verbose=True):
    """
    Common evaluation function that runs the simulation, applies noise if specified,
    predicts errors using the model, applies corrections, and optionally plots the results.

    Parameters:
    - model: The trained model.
    - base_configurations: Base configurations for the simulation.
    - common_parameters: Common parameters for the simulation.
    - dataset_scalers: Dictionary containing 'input_scaler' and 'target_scaler'.
    - noise_type: BPM noise or quad_tilt noise. 'bpm', 'quad_tilt'
    - noise_level: The level of noise to add to BPM readings. If 0, no noise is added.
    - plot: Boolean flag to control plotting. True for evaluate_once(), False for benchmarking.
    - verbose: Boolean flag to control print statements.
    
    Returns:
    - actual_deltas: Dictionary of actual quadrupole error deltas.
    - predicted_deltas: Dictionary of predicted quadrupole error deltas.
    - residual_error: Mean absolute residual error after correction.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # * Define a vertical quadrupole error
    mean_or_min_delta, std_or_max_delta = common_parameters['delta_range']
    # Define quad tilt errors
    mean_or_min_quad_tilt_error, std_or_max_quad_tilt_error = common_parameters['quad_tilt_angle_range']
    # Define dipole tilt errors
    mean_or_min_dipole_tilt_error, std_or_max_dipole_tilt_error = common_parameters['dipole_tilt_angle_range']
    
    
    # quad_tilt Noise level
    # for benchmarking quad tilts, we sample from normal distribution of a larger std
    if noise_type == 'quad_tilt' and noise_level > 0:
        print(f"Applying random noise to Quadrupole Tilt angles ±{noise_level}.")
        mean_or_min_quad_tilt_error, std_or_max_quad_tilt_error = 0.0, noise_level
        
    
    # * Prepare the configuration for evaluation
    eval_config = copy.deepcopy(base_configurations[0])
    
    
    if eval_config['quad_errors']:
        fodo_indices_with_error = [err['FODO_index'] for err in eval_config['quad_errors']]
    else:
        fodo_indices_with_error = []
    
    if eval_config['quad_tilt_errors']:
        fodo_indices_with_quad_tilt_error = [err['FODO_index'] for err in eval_config['quad_tilt_errors']]
    else:
        fodo_indices_with_quad_tilt_error = []
    
    if eval_config['dipole_tilt_errors']:
        fodo_indices_with_dipole_tilt_error = [err['FODO_index'] for err in eval_config['dipole_tilt_errors']]
    else:
        fodo_indices_with_dipole_tilt_error = []
    
    
    sampling_func = None
    if common_parameters['random_criterion'] == 'uniform':
        sampling_func = np.random.uniform
    elif common_parameters['random_criterion'] == 'normal':
        sampling_func = np.random.normal
    
    
    
    # Target prediction (Misalignment)
    quadrupole_errors_target_values = {}
    
    # Apply quad misalignment errors
    if eval_config['quad_errors']:
        for qe_ix, qe in enumerate(eval_config['quad_errors']):
            if eval_config['quad_errors'][qe_ix]['FODO_index'] in fodo_indices_with_error:
                quadrupole_error_delta = sampling_func(mean_or_min_delta, std_or_max_delta)
                quadrupole_errors_target_values[qe_ix] = quadrupole_error_delta
                eval_config['quad_errors'][qe_ix]['delta'] = quadrupole_error_delta
                if verbose:
                    print("_run_evaluation()/ ", qe_ix, qe)
            else:
                eval_config['quad_errors'][qe_ix]['delta'] = 0.0

    # Apply quad_tilt_errors
    if eval_config['quad_tilt_errors']:
        for qe_ix, qe in enumerate(eval_config['quad_tilt_errors']):
            if eval_config['quad_tilt_errors'][qe_ix]['FODO_index'] in fodo_indices_with_quad_tilt_error:
                quadrupole_tilt_error_delta = sampling_func(mean_or_min_quad_tilt_error, std_or_max_quad_tilt_error)
                print("quadrupole_tilt_error_delta: ", quadrupole_tilt_error_delta)
                
                # TODO(aribra): # Target prediction (Quadrupole Tilt)
                # quadrupole_errors_target_values[qe_ix] = quadrupole_tilt_error_delta
                
                eval_config['quad_tilt_errors'][qe_ix]['tilt_angle'] = quadrupole_tilt_error_delta
                if verbose:
                    print("_run_evaluation()/ ", qe_ix, qe)
            else:
                eval_config['quad_tilt_errors'][qe_ix]['tilt_angle'] = 0.0

    # Apply dipole_tilt_errors
    if eval_config['dipole_tilt_errors']:
        for qe_ix, qe in enumerate(eval_config['dipole_tilt_errors']):
            if eval_config['dipole_tilt_errors'][qe_ix]['FODO_index'] in fodo_indices_with_dipole_tilt_error:
                dipole_tilt_error_delta = sampling_func(mean_or_min_dipole_tilt_error, std_or_max_dipole_tilt_error)
                
                # TODO(aribra): # Target prediction (Dipole Tilt)
                # quadrupole_errors_target_values[qe_ix] = dipole_tilt_error_delta
                
                eval_config['dipole_tilt_errors'][qe_ix]['tilt_angle'] = dipole_tilt_error_delta
                if verbose:
                    print("_run_evaluation()/ ", qe_ix, qe)
            else:
                eval_config['dipole_tilt_errors'][qe_ix]['tilt_angle'] = 0.0
            
                

    if verbose:
        print("evaluate_model()/ base_configurations: ", eval_config)

    # * Simulate without error (baseline) + after applying the error
    sim_runner = SimulationRunner(
        base_configurations=[eval_config],
        common_parameters=common_parameters
    )

    initial_states = None

    sim_runner.run_configurations(draw_plots=False, verbose=verbose, initial_states=initial_states)

    initial_states = sim_runner.initial_states

    simulator_no_error = sim_runner.simulators_no_error.get(f"{eval_config['config_name']} - No Error")
    simulator_with_error = sim_runner.simulators_with_error.get(f"{eval_config['config_name']} - With Error")

    # * The initial_states are the same in both simulations

    merged_config = {**common_parameters, **eval_config}

    if noise_type == 'bpm' and noise_level > 0:
        if verbose:
            print(f"Applying random noise to BPM readings in X and Y axis ±{noise_level}.")
        simulator_with_error.bpm_readings['x'] = simulator_with_error.bpm_readings['x'] + \
            np.random.uniform(-noise_level, noise_level, simulator_with_error.bpm_readings['x'].shape)
        simulator_with_error.bpm_readings['y'] = simulator_with_error.bpm_readings['y'] + \
            np.random.uniform(-noise_level, noise_level, simulator_with_error.bpm_readings['y'].shape)
    
    
    # Create SimulationDataset instance
    simulation_dataset = SimulationDataset(
        merged_config=merged_config,
        bpm_readings_no_error=simulator_no_error.bpm_readings,
        bpm_readings_with_error=simulator_with_error.bpm_readings,
        bpm_positions=simulator_no_error.bpm_positions,
        quadrupole_errors=simulator_with_error.quad_errors,
        quadrupole_tilt_errors=simulator_with_error.quadrupole_tilt_errors,
        dipole_tilt_errors=simulator_with_error.dipole_tilt_errors,
        lattice_reference=simulator_no_error.get_lattice_reference(),
        apply_avg=common_parameters.get('apply_avg', False)
    )

    # Generate data using the same parameters as during training
    start_rev = common_parameters.get('start_rev', 0)
    end_rev = common_parameters.get('end_rev', simulator_no_error.n_turns)
    fodo_cell_indices = common_parameters.get('fodo_cell_indices', list(range(simulator_no_error.n_FODO)))
    planes = common_parameters.get('planes', ['x', 'y'])

    if verbose:
        print(f"[Evaluation - Generate data params:]\n"
              f"\t start_rev={start_rev}\n"
              f"\t end_rev={end_rev}\n"
              f"\t fodo_cell_indices={fodo_cell_indices}\n"
              f"\t planes={planes}")
    

    (input_tensor, target_tensor,
     error_values_quad_misalign,
     error_values_quad_tilt,
     error_values_dipole_tilt) = simulation_dataset.generate_data(
        start_rev, end_rev, fodo_cell_indices, planes
    )

    # Reshape input data to match model input
    n_samples, n_turns, n_BPMs, n_planes = input_tensor.shape
    input_size = n_BPMs * n_planes
    input_data = input_tensor.reshape(n_samples, n_turns, input_size)

    # **Reshape input data back to (n_samples, n_turns, n_BPMs, n_planes)**
    input_data_reshaped = input_data.reshape(n_samples, n_turns, n_BPMs, n_planes)

    # **Reshape to (-1, n_planes) for scaling**
    input_data_flat = input_data_reshaped.reshape(-1, n_planes)  # Shape: (n_samples * n_turns * n_BPMs, n_planes)

    # **Use the input scaler to transform the input data**
    input_data_flat_scaled = dataset_scalers['input_scaler'].transform(input_data_flat)

    # **Reshape back to (n_samples, n_turns, n_BPMs, n_planes)**
    input_data_scaled = input_data_flat_scaled.reshape(n_samples, n_turns, n_BPMs, n_planes)

    # **Flatten to (n_samples, n_turns, input_size) for model input**
    input_data_scaled = input_data_scaled.reshape(n_samples, n_turns, input_size)

    # Convert to tensor
    input_tensor_model = torch.tensor(input_data_scaled, dtype=torch.float32).to(device)


    # Predict the error
    model.eval()
    with torch.no_grad():
        if verbose:
            print("input to model: ", input_tensor_model.shape)
        predicted_error = model(input_tensor_model)  # predicted_error shape: (n_samples, output_size)
        if verbose:
            print("predicted_errors = ", predicted_error)
        predicted_errors_scaled_values = predicted_error.cpu().numpy()

    # Inverse transform the prediction
    predicted_errors_values_transformed_back = dataset_scalers['target_scaler'].inverse_transform(predicted_errors_scaled_values)
    predicted_errors_values_transformed_back = predicted_errors_values_transformed_back.flatten()
    if verbose:
        print(f"predicted_errors_values_transformed_back = {predicted_errors_values_transformed_back}")
    
    actual_deltas = {}
    predicted_deltas = {}
    
    for pesv_ix, pesv in enumerate(predicted_errors_values_transformed_back):
        predicted_error_value = pesv
        predicted_deltas[pesv_ix] = predicted_error_value
        if pesv_ix not in quadrupole_errors_target_values:
            if verbose:
                print(f"WARNING - error prediction output with index={pesv_ix}, is not available.\n\tthis may indicate that you set custom error config rather than the trained network")
            continue
        actual_deltas[pesv_ix] = quadrupole_errors_target_values[pesv_ix]
        if verbose:
            print(f"\tActual quadrupole error delta: {quadrupole_errors_target_values[pesv_ix]:.7e}, {quadrupole_errors_target_values[pesv_ix] * 1e6}")
            print(f"\tPredicted quadrupole error delta: {predicted_error_value:.7e}, {predicted_error_value * 1e6}")
            print('---')

    # * Apply correction and re-run simulation
    # Correct the quadrupole error by subtracting the predicted error
    corrected_deltas = {}
    for pevtb_ix, pevtb in predicted_deltas.items():
        if pevtb_ix not in quadrupole_errors_target_values:
            if verbose:
                print(f"WARNING - error prediction output with index={pevtb_ix}, is not available.\n\tthis may indicate that you set custom error config rather than the trained network")
            # If error was not found, we assume it is 0
            corrected_deltas[pevtb_ix] = 0.0
            continue
        corrected_delta = quadrupole_errors_target_values[pevtb_ix] - pevtb
        corrected_deltas[pevtb_ix] = corrected_delta
        if verbose:
            print(f"\tCorrected_delta quadrupole error delta [0]: {corrected_delta:.7e}, {corrected_delta * 1e6}")
    
    if merged_config['target_data'] == 'quad_misalign_deltas':
        target_errors_key = 'quad_errors'
    elif merged_config['target_data'] == 'quad_tilt_angles':
        target_errors_key = 'quad_tilt_errors'
    elif merged_config['target_data'] == 'dipole_tilt_angles':
        target_errors_key = 'dipole_tilt_errors'
    
    eval_config_corrected = eval_config.copy()
    for cord_ix, cord in corrected_deltas.items():
        if cord_ix < len(eval_config_corrected[target_errors_key]):
            eval_config_corrected[target_errors_key][cord_ix]['delta'] = cord
            if verbose:
                print(cord_ix, cord)

    runner_corrected = SimulationRunner(
        base_configurations=[eval_config_corrected],
        common_parameters=common_parameters
    )

    runner_corrected.run_configurations(draw_plots=False, verbose=False, initial_states=initial_states)
    simulator_corrected = runner_corrected.simulators_with_error.get(f"{eval_config_corrected['config_name']} - With Error")

    # * Compare y positions after applying the correction with the original simulation without errors
    # Use the same start_rev and end_rev


    if plot:
        # Extract BPM readings for comparison
        bpm_readings_no_error = simulator_no_error.bpm_readings['y'][:, start_rev:end_rev, :][:, :, fodo_cell_indices].mean(axis=0)  # Shape: [n_turns, n_BPMs]
        bpm_readings_with_error = simulator_with_error.bpm_readings['y'][:, start_rev:end_rev, :][:, :, fodo_cell_indices].mean(axis=0)  # Shape: [n_turns, n_BPMs]
        bpm_readings_corrected = simulator_corrected.bpm_readings['y'][:, start_rev:end_rev, :][:, :, fodo_cell_indices].mean(axis=0)  # Shape: [n_turns, n_BPMs]
    
        bpm_indx = 3
        rev_numbers = np.arange(end_rev - 100, end_rev)
        if verbose:
            print(f"rev_numbers = {rev_numbers}")
            print(f"bpm_readings_no_error = {bpm_readings_no_error.shape}")
        # Plot the comparison
        plt.figure(figsize=(12, 6))
        plt.plot(rev_numbers, bpm_readings_no_error[-100:, bpm_indx], '-o', label='No Error', color='blue')
        plt.plot(rev_numbers, bpm_readings_with_error[-100:, bpm_indx], '-x', label='With Error', color='red')
        plt.plot(rev_numbers, bpm_readings_corrected[-100:, bpm_indx], '-v', label='After Correction', color='green')
        plt.xlabel('Turn')
        plt.ylabel(f'Average y position at BPM {bpm_indx}')
        plt.title('Comparison of y positions after correction')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Compute and print the residual error after correction
        residual_error = np.abs(bpm_readings_no_error - bpm_readings_corrected).mean()
        print(f"Residual error after correction (mean absolute difference): {residual_error:.6e}, {residual_error * 1e6}")
        
        # Additional plots
        simulator_no_error.plot_comparison(simulator_with_error, cell_idx=bpm_indx, viz_start_idx=end_rev - 100, 
                                             viz_end_idx=end_rev, save_label="WOEvsWE", window_size=50, plot_all=True, extra_title="Before correction")
    
        simulator_no_error.plot_comparison(simulator_corrected, cell_idx=bpm_indx, 
                                            viz_start_idx=end_rev - 100, viz_end_idx=end_rev, save_label="WOEvsC", window_size=50, plot_all=True, extra_title="After correction")
    
        simulator_no_error.plot_bpm_heatmaps(cell_idx=bpm_indx, simulation_label='No Error')
        simulator_with_error.plot_bpm_heatmaps(cell_idx=bpm_indx, simulation_label='With Error')
        simulator_corrected.plot_bpm_heatmaps(cell_idx=bpm_indx, simulation_label='Corrected')
    
        simulator_corrected.plot_bpm_comparison_last_images(simulator_corrected, simulator_with_error, cell_idx=bpm_indx, save_label="WEvsC", particles='all')
        simulator_corrected.plot_bpm_comparison_last_images(simulator_corrected, simulator_no_error, cell_idx=bpm_indx, save_label="WOEvsC", particles='all')

    return actual_deltas, predicted_deltas


def evaluate_once(model, base_configurations, common_parameters, dataset_scalers):
    """
    Performs a single evaluation without adding noise.
    Maintains all original prints and plots.

    Parameters:
    - model: The trained model.
    - base_configurations: Base configurations for the simulation.
    - common_parameters: Common parameters for the simulation.
    - dataset_scalers: Dictionary containing 'input_scaler' and 'target_scaler'.
    
    Returns:
    - None
    """
    _run_evaluation(
        model=model,
        base_configurations=base_configurations,
        common_parameters=common_parameters,
        dataset_scalers=dataset_scalers,
        noise_type=None, 
        noise_level=0.0,
        plot=True,
        verbose=True
    )


def benchmark_evaluation_bpm_noise(model, base_configurations, common_parameters, dataset_scalers, noise_start=0, noise_stop=100e-6, bins=11, runs=20):
    """
    Automates running the evaluation with varying levels of BPM reading noise.
    Collects statistics and plots model prediction accuracy against noise levels.

    Parameters:
    - model: The trained model.
    - base_configurations: Base configurations for the simulation.
    - common_parameters: Common parameters for the simulation.
    - dataset_scalers: Dictionary containing 'input_scaler' and 'target_scaler'.
    - noise_start: Minimum noise level to apply.
    - noise_stop: Maximum noise level to apply.
    - bins: Number of bins for noise levels.
    - runs: Number of evaluations to run per noise level.
    
    Returns:
    - stats: Dictionary containing statistics for each noise level and FODO index.
    """
    
    noise_pallette = np.linspace(noise_start, noise_stop, bins)  # Convert to meters if needed

    # Initialize a dictionary to store statistics
    stats = defaultdict(lambda: defaultdict(list))  # stats[noise_level][fodo_index] = list of errors

    for noise_level in noise_pallette:
        print("Running evaluation for noise_level=", noise_level)
        if model.training:
            model.eval()  # Ensure model is in evaluation mode
        for run in range(runs):
            print(f"\t-------------[Run {run + 1}/{runs}]")
            actual_deltas, predicted_deltas = _run_evaluation(
                model=model,
                base_configurations=base_configurations,
                common_parameters=common_parameters,
                dataset_scalers=dataset_scalers,
                noise_type='bpm',
                noise_level=noise_level,
                plot=False,
                verbose=False
            )
            for fodo_ix in actual_deltas:
                error = np.abs(actual_deltas[fodo_ix] - predicted_deltas[fodo_ix])
                stats[noise_level][fodo_ix].append(error)
        print(f"Completed benchmarking for noise_level={noise_level} meters.")

    return stats


def benchmark_evaluation_tilt_noise(model, base_configurations, common_parameters, dataset_scalers, noise_start=10, noise_stop=50, bins=5, runs=50):
    """
    Automates running the evaluation with varying levels of BPM reading noise.
    Collects statistics and plots model prediction accuracy against noise levels.

    Parameters:
    - model: The trained model.
    - base_configurations: Base configurations for the simulation.
    - common_parameters: Common parameters for the simulation.
    - dataset_scalers: Dictionary containing 'input_scaler' and 'target_scaler'.
    - noise_start: Minimum noise level to apply.
    - noise_stop: Maximum noise level to apply.
    - bins: Number of bins for noise levels.
    - runs: Number of evaluations to run per noise level.
    
    Returns:
    - stats: Dictionary containing statistics for each noise level and FODO index.
    """
    
    noise_pallette = np.linspace(noise_start, noise_stop, bins)

    # Initialize a dictionary to store statistics
    stats = defaultdict(lambda: defaultdict(list))  # stats[noise_level][fodo_index] = list of errors

    for noise_level in noise_pallette:
        print("Running evaluation for noise_level=", noise_level)
        if model.training:
            model.eval()  # Ensure model is in evaluation mode
        for run in range(runs):
            print(f"\t-------------[Run {run + 1}/{runs}]")
            actual_deltas, predicted_deltas = _run_evaluation(
                model=model,
                base_configurations=base_configurations,
                common_parameters=common_parameters,
                dataset_scalers=dataset_scalers,
                noise_type='quad_tilt',
                noise_level=noise_level,
                plot=False,
                verbose=False
            )
            for fodo_ix in actual_deltas:
                error = np.abs(actual_deltas[fodo_ix] - predicted_deltas[fodo_ix])
                stats[noise_level][fodo_ix].append(error)
        print(f"Completed benchmarking for noise_level={noise_level} meters.")

    return stats


def split_merged_config(merged_config):
    # Keys that belong to base_configurations
    base_keys = {
        'config_name',
        'design_radius',
        'n_FODO',
        'f',
        'L_quad',
        'L_straight',
        'quad_errors',
        'quad_tilt_errors',
        'dipole_tilt_errors',
        'total_dipole_bending_angle'
    }

    # Extract base configuration parameters
    base_config = {k: v for k, v in merged_config.items() if k in base_keys}

    # Wrap the base_config in a list as
    base_configurations = [base_config]

    # Extract common parameters by excluding base_keys
    common_parameters = {k: v for k, v in merged_config.items() if k not in base_keys}

    return base_configurations, common_parameters


def main_evaluation_block(model, data_sub_cfg, benchmark_type=None, run_benchmark=False):
    '''
    benchmark_type: 'bpm' or 'quad_tilt'
    '''
    
    merged_config = data_sub_cfg['merged_config']
    input_scaler_config = data_sub_cfg['input_scaler_config']
    target_scaler_config = data_sub_cfg['target_scaler_config']
    overridden_base_config = data_sub_cfg['overridden_base_config']
    
    # Data scalers
    dataset_scalers = {
        'input_scaler': deserialize_minmax_scaler(input_scaler_config),
        'target_scaler': deserialize_minmax_scaler(target_scaler_config)
    }

    # Flags for evaluation mode
    run_evaluate_once = True
    if run_benchmark and benchmark_type is not None:
        run_evaluate_once = False
    
    if benchmark_type == 'bpm':
        print("Running benchmark for BPM noise...")

        # Parameters for benchmarking BPM noise
        NOISE_START = 0
        NOISE_STOP = 100e-6  # meters
        BINS = 11
        NOISE_PALLETTE = np.linspace(NOISE_START, NOISE_STOP, BINS)  # Converted to micro-units for consistency with example
        # Number of runs per noise level
        RUNS_PER_NOISE = 50

    elif benchmark_type == 'quad_tilt':
        print("Running benchmark for Quadrupole Tilt noise...")
        
        # Parameters for benchmarking quad_tilt
        NOISE_START = 0.01  # mrads
        NOISE_STOP = 0.05  # mrads
        BINS = 5
        NOISE_PALLETTE = np.linspace(NOISE_START, NOISE_STOP, BINS)  # Converted to micro-units for consistency with example
        # Number of runs per noise level
        RUNS_PER_NOISE = 50

    CANCEL_TILT_ERROR = False
    CANCEL_MISALIGN_ERROR = False

    base_configurations, common_parameters = split_merged_config(merged_config)
    base_configurations = overridden_base_config

    # Set up FODO mapping dictionary
    # This dictionary maps each prediction output index to the corresponding FODO indices where errors were introduced.
    # The target data preparation and network output are ordered according to the configured quad_errors.
    fodo_mapping = {}
    
    # Extract FODO cell indices
    if merged_config['target_data'] == 'quad_misalign_deltas':
        target_errors_cfg = merged_config['quad_errors']
        
    elif merged_config['target_data'] == 'quad_tilt_angles':
        target_errors_cfg = merged_config['quad_tilt_errors']
        
    elif merged_config['target_data'] == 'dipole_tilt_angles':
        target_errors_cfg = merged_config['dipole_tilt_errors']
    
    for qe_ix, qe in enumerate(target_errors_cfg):
        fodo_mapping[qe_ix] = qe['FODO_index']


    benchmark_info = {
        "benchmark_type": benchmark_type,
        "noise_start": NOISE_START,
        "noise_stop": NOISE_STOP,
        "bins": BINS,
        "noise_pallette": NOISE_PALLETTE,
        "runs_per_noise": RUNS_PER_NOISE,
        "fodo_mapping": fodo_mapping,
        "cancel_tilt_error": CANCEL_TILT_ERROR,
        "cancel_misalign_error": CANCEL_MISALIGN_ERROR
    }

    if CANCEL_TILT_ERROR:
        base_configurations['quad_tilt_errors'] = []
        base_configurations['dipole_tilt_errors'] = []

    if CANCEL_MISALIGN_ERROR:
        base_configurations['quad_errors'] = []

    if CANCEL_TILT_ERROR and CANCEL_MISALIGN_ERROR:
        common_parameters['target_data'] = False

    common_parameters['num_particles'] = 10


    if run_evaluate_once:
        evaluate_once(model, [base_configurations], common_parameters, dataset_scalers)

    elif run_benchmark:
        if CANCEL_MISALIGN_ERROR and CANCEL_TILT_ERROR:
            print("BENCHMARK was not run!!")
        
        elif benchmark_type == 'bpm':
            print("Running bpm benchmark...")
            stats = benchmark_evaluation_bpm_noise(
                model=model,
                base_configurations=[base_configurations],
                common_parameters=common_parameters,
                dataset_scalers=dataset_scalers,
                noise_start=NOISE_START,
                noise_stop=NOISE_STOP,                
                bins=BINS,
                runs=RUNS_PER_NOISE
            )
            
            stats['benchmark_info'] = benchmark_info
            
            save_stats_path = f"{SAVE_DIR_BENCHMARKS}/benchmark_stats_bpm_MisAlign-True_Tilt_True.pt"
            
            print(f"save_stats_path: {save_stats_path}")
    
            torch.save(convert_defaultdict_to_dict(stats), save_stats_path)

            plot_benchmark_stats(stats, benchmark_info)

        elif benchmark_type == 'quad_tilt':
            print("Running quad_tilt benchmark...")
            stats = benchmark_evaluation_tilt_noise(
                model=model,
                base_configurations=[base_configurations],
                common_parameters=common_parameters,
                dataset_scalers=dataset_scalers,
                noise_start=NOISE_START,
                noise_stop=NOISE_STOP,
                bins=BINS,
                runs=RUNS_PER_NOISE
            )

            stats['benchmark_info'] = benchmark_info

            save_stats_path = f"{SAVE_DIR_BENCHMARKS}/benchmark_stats_quad_tilt_MisAlign-True_Tilt-True.pt"

            print(f"save_stats_path: {save_stats_path}")

            torch.save(convert_defaultdict_to_dict(stats), save_stats_path)

            plot_benchmark_stats(stats, benchmark_info)

