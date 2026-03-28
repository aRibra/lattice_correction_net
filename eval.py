# eval.py

import copy
import math
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
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

    print(".....[[Running inference_on_validation_data]].....")

    print("dataset_scalers['target_scaler']------ ", dataset_scalers["target_scaler"])
    print(dataset_scalers["target_scaler"].data_range_)
    print(dataset_scalers["target_scaler"].data_max_)
    print(dataset_scalers["target_scaler"].data_min_)
    print(dataset_scalers["target_scaler"].scale_)
    print(dataset_scalers["target_scaler"].min_)

    # Extract FODO cell indices
    if merged_config["target_data"] == "quad_misalign_deltas":
        fodo_cell_indices = [err["FODO_index"] for err in merged_config["quad_errors"]]
    elif merged_config["target_data"] == "quad_tilt_angles":
        fodo_cell_indices = [
            err["FODO_index"] for err in merged_config["quad_tilt_errors"]
        ]
    elif merged_config["target_data"] == "dipole_tilt_angles":
        fodo_cell_indices = [
            err["FODO_index"] for err in merged_config["dipole_tilt_errors"]
        ]

    print(fodo_cell_indices)

    # Batch parameters
    batch_limit_s = 0
    batch_limit_e = 16
    nb_batches = 6
    batch_counter = 0

    # Number of columns for subplots
    cols = 3
    rows = math.ceil(
        len(fodo_cell_indices) / cols
    )  # Calculate the number of rows required

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
        # Apply inverse transform and convert to microns
        output_scaled = (
            dataset_scalers["target_scaler"].inverse_transform(output.cpu().numpy())
            * 1e6
        )
        # Apply inverse transform and convert to microns
        batch_targets_scaled = (
            dataset_scalers["target_scaler"].inverse_transform(
                batch_targets.cpu().numpy()
            )
            * 1e6
        )

        # Calculate residuals
        err_resid = output_scaled - batch_targets_scaled[batch_limit_s:batch_limit_e]

        # Create the main figure
        fig = plt.figure(figsize=(20, 6 * rows))  # Increased height for larger plots
        # Create a top-level GridSpec with reduced spacing
        main_gs = GridSpec(rows, cols, figure=fig, wspace=0.3, hspace=0.3)

        # Supertitle
        fig.suptitle(
            "Prediction Samples Vs Ground Truth on Validation Data",
            fontsize=20,  # Adjust font size as needed
            y=0.95,  # Adjust y-position to prevent overlap
            fontweight="bold",  # Optional: Make the title bold
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
                batch_targets_scaled[:, quad_idx_pred],
                "-gs",
                lw=5,  # Increased line width for better visibility
                alpha=0.5,
                label="Ground Truth",
            )
            main_ax.plot(
                output_scaled[:, quad_idx_pred], "-.b", lw=2, label="Prediction"
            )
            main_ax.plot(
                err_resid[:, quad_idx_pred], "-or", lw=1, label="Residual Error"
            )
            main_ax.legend(["gt", "pred", "err_resid"], fontsize=14)
            main_ax.set_title(
                f"FODO Cell Index: {fodo_cell_indices[quad_idx_pred]}", fontsize=18
            )
            main_ax.set_ylabel("Predicted Error\n(µm)", fontsize=18)
            main_ax.tick_params(axis="both", labelsize=15)  # Set font size for ticks
            main_ax.minorticks_on()

            # Plotting on the residual axes
            resid_ax.plot(
                err_resid[:, quad_idx_pred], "-or", lw=1, label="Residual Error"
            )
            resid_ax.legend(["Residual"], fontsize=14)
            resid_ax.set_ylabel("Residual\n(µm)", fontsize=18)
            resid_ax.set_xlabel("Batch Sample", fontsize=15)
            resid_ax.tick_params(axis="both", labelsize=15)  # Set font size for ticks
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
                empty_ax.axis("off")

        plt.tight_layout()
        plt.savefig(
            f"{SAVE_DIR_FIGS}/inference_val_batch_{batch_counter}.eps",
            bbox_inches="tight",
            format="eps",
        )
        plt.show()


def evaluate_with_existing_data(
    model,
    val_loader,
    dataset_scalers,
    merged_config,
    noise_type="bpm_shift",
    noise_level=0.0,
    x_shift=True,
    y_shift=True,
    verbose=False,
):
    """
    Evaluates the model on existing validation data with systematic BPM shifts applied in batches.

    Parameters:
    - model: The trained model.
    - val_loader: DataLoader for the validation dataset.
    - dataset_scalers: Dictionary containing 'input_scaler' and 'target_scaler'.
    - merged_config: Merged configuration dictionary.
    - noise_type: Type of noise to apply ('bpm_shift').
    - noise_level: The level of systematic shift to apply to BPM readings (in meters).
    - x_shift: Whether to apply shift on X-axis.
    - y_shift: Whether to apply shift on Y-axis.
    - verbose: Boolean flag to control print statements.

    Returns:
    - actual_deltas: Dictionary of actual quadrupole error deltas.
    - predicted_deltas: Dictionary of predicted quadrupole error deltas.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Extract FODO cell indices from merged_config
    if merged_config["target_data"] == "quad_misalign_deltas":
        target_errors_key = "quad_errors"
    elif merged_config["target_data"] == "quad_tilt_angles":
        target_errors_key = "quad_tilt_errors"
    elif merged_config["target_data"] == "dipole_tilt_angles":
        target_errors_key = "dipole_tilt_errors"
    fodo_cell_indices = [err["FODO_index"] for err in merged_config[target_errors_key]]

    actual_deltas = {}
    predicted_deltas = {}

    for batch_inputs, batch_targets in val_loader:
        # Apply systematic BPM shifts
        if noise_type == "bpm_shift" and noise_level != 0:
            shift_axes = []
            if x_shift:
                shift_axes.append("x")
            if y_shift:
                shift_axes.append("y")
            if verbose:
                print(
                    f"Applying systematic shift of {noise_level * 1e6:.1f}μm to BPM readings on axes: {shift_axes}"
                )

            # Reshape inputs to (batch_size, n_turns, n_BPMs, n_planes)
            batch_size, n_turns, input_size = batch_inputs.shape
            # TODO(aribra): support only x | y
            if input_size == 1:
                raise Exception(
                    "[evaluate_with_existing_data()] still does not support evaluating on single plane."
                )
            n_BPMs = input_size // 2  # Assuming 2 planes (x, y)
            batch_inputs = batch_inputs.reshape(batch_size, n_turns, n_BPMs, 2)

            # Apply shifts
            for axis, plane_idx in [("x", 0), ("y", 1)]:
                if axis in shift_axes:
                    batch_inputs[:, :, :, plane_idx] += noise_level
                    if verbose:
                        print(
                            f"  Applied {noise_level * 1e6:.1f}μm shift to {axis}-axis"
                        )

            bpm_noise_level = 1e-6
            batch_inputs[:, :, :, 0] = batch_inputs[:, :, :, 0] + np.random.normal(
                0, bpm_noise_level, batch_inputs[:, :, :, 0].shape
            )
            batch_inputs[:, :, :, 1] = batch_inputs[:, :, :, 1] + np.random.normal(
                0, bpm_noise_level, batch_inputs[:, :, :, 1].shape
            )

            # Reshape back to (batch_size, n_turns, input_size)
            batch_inputs = batch_inputs.reshape(batch_size, n_turns, input_size)

        # if noise_type == 'bpm' and noise_level > 0:
        # if verbose:
        # print(f"Applying random noise to BPM readings in X and Y axis with std={noise_level}.")

        # if noise_type == 'bpm' and noise_level > 0:
        # if verbose:
        # print(f"Applying random noise to BPM readings in X and Y axis with std={noise_level}.")

        # Perform inference
        with torch.no_grad():
            batch_inputs = batch_inputs.to(device)
            predicted_errors = model(batch_inputs)
        predicted_errors = predicted_errors.cpu().numpy()

        # Inverse transform predictions and targets
        predicted_errors_transformed = dataset_scalers[
            "target_scaler"
        ].inverse_transform(predicted_errors)
        batch_targets_transformed = dataset_scalers["target_scaler"].inverse_transform(
            batch_targets
        )

        # Store actual and predicted deltas
        for batch_idx in range(batch_targets.shape[0]):
            for fodo_idx, quad_idx in enumerate(fodo_cell_indices):
                sample_idx = len(actual_deltas)  # Unique index for each sample
                actual_deltas[sample_idx] = batch_targets_transformed[
                    batch_idx, fodo_idx
                ]
                predicted_deltas[sample_idx] = predicted_errors_transformed[
                    batch_idx, fodo_idx
                ]
                if verbose:
                    print(f"Sample {sample_idx}, FODO {quad_idx}:")
                    print(
                        f"\tActual delta: {actual_deltas[sample_idx]:.7e}, {actual_deltas[sample_idx] * 1e6}μm"
                    )
                    print(
                        f"\tPredicted delta: {predicted_deltas[sample_idx]:.7e}, {predicted_deltas[sample_idx] * 1e6}μm"
                    )
                    print("---")

    return actual_deltas, predicted_deltas


def _run_evaluation(
    model,
    base_configurations,
    common_parameters,
    dataset_scalers,
    noise_type=None,
    noise_level=0.0,
    x_shift=False,
    y_shift=False,
    plot=False,
    verbose=True,
    k_errors_config=None,
    include_quad_tilt=False,
    quad_tilt_range=(0.01, 0.05),
    include_bpm_noise=False,
    bpm_noise_range=(0, 100e-6),
):
    """
    Common evaluation function that runs the simulation, applies noise if specified,
    predicts errors using the model, applies corrections, and optionally plots the results.

    Parameters:
    - model: The trained model.
    - base_configurations: Base configurations for the simulation.
    - common_parameters: Common parameters for the simulation.
    - dataset_scalers: Dictionary containing 'input_scaler' and 'target_scaler'.
    - noise_type: BPM noise, quad_tilt noise, or k_errors. 'bpm', 'quad_tilt', 'k_errors'
    - noise_level: The level of noise to add. If 0, no noise is added.
    - x_shift: Whether to apply shift on X-axis (only for noise_type='bpm_shift')
    - y_shift: Whether to apply shift on Y-axis (only for noise_type='bpm_shift')
    - plot: Boolean flag to control plotting. True for evaluate_once(), False for benchmarking.
    - verbose: Boolean flag to control print statements.
    - k_errors_config: K errors configuration dict.
    - include_quad_tilt: Include quad tilt as additional error source.
    - quad_tilt_range: Quad tilt range in mrads.
    - include_bpm_noise: Include BPM noise as additional error source.
    - bpm_noise_range: BPM noise range in meters.

    Returns:
    - actual_deltas: Dictionary of actual quadrupole error deltas.
    - predicted_deltas: Dictionary of predicted quadrupole error deltas.
    - residual_error: Mean absolute residual error after correction.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # * Define a vertical quadrupole error
    mean_or_min_delta, std_or_max_delta = common_parameters["delta_range"]
    # Define quad tilt errors
    mean_or_min_quad_tilt_error, std_or_max_quad_tilt_error = common_parameters[
        "quad_tilt_angle_range"
    ]
    # Define dipole tilt errors
    mean_or_min_dipole_tilt_error, std_or_max_dipole_tilt_error = common_parameters[
        "dipole_tilt_angle_range"
    ]

    # quad_tilt Noise level
    # for benchmarking quad tilts, we sample from normal distribution of a larger std
    if noise_type == "quad_tilt" and noise_level > 0:
        print(f"Applying random noise to Quadrupole Tilt angles ±{noise_level}.")
        mean_or_min_quad_tilt_error, std_or_max_quad_tilt_error = 0.0, noise_level

    # * Prepare the configuration for evaluation
    eval_config = copy.deepcopy(base_configurations[0])

    # Insert k_errors into eval_config if provided
    if k_errors_config is not None:
        eval_config["k_errors"] = k_errors_config

    # Handle additional error sources (quad_tilt)
    if include_quad_tilt:
        mean_or_min_quad_tilt_error, std_or_max_quad_tilt_error = (
            quad_tilt_range[0],
            quad_tilt_range[1],
        )

    # Handle additional error sources (bpm_noise) - BPM noise is applied after simulation
    bpm_noise_value = 0.0
    if include_bpm_noise:
        bpm_noise_value = np.random.uniform(bpm_noise_range[0], bpm_noise_range[1])

    if eval_config["quad_errors"]:
        fodo_indices_with_error = [
            err["FODO_index"] for err in eval_config["quad_errors"]
        ]
    else:
        fodo_indices_with_error = []

    if eval_config["quad_tilt_errors"]:
        fodo_indices_with_quad_tilt_error = [
            err["FODO_index"] for err in eval_config["quad_tilt_errors"]
        ]
    else:
        fodo_indices_with_quad_tilt_error = []

    if eval_config["dipole_tilt_errors"]:
        fodo_indices_with_dipole_tilt_error = [
            err["FODO_index"] for err in eval_config["dipole_tilt_errors"]
        ]
    else:
        fodo_indices_with_dipole_tilt_error = []

    sampling_func = None
    if common_parameters["random_criterion"] == "uniform":
        sampling_func = np.random.uniform
    elif common_parameters["random_criterion"] == "normal":
        sampling_func = np.random.normal

    # Target prediction (Misalignment)
    quadrupole_errors_target_values = {}

    # Apply quad misalignment errors
    if eval_config["quad_errors"]:
        for qe_ix, qe in enumerate(eval_config["quad_errors"]):
            if (
                eval_config["quad_errors"][qe_ix]["FODO_index"]
                in fodo_indices_with_error
            ):
                quadrupole_error_delta = sampling_func(
                    mean_or_min_delta, std_or_max_delta
                )
                quadrupole_errors_target_values[qe_ix] = quadrupole_error_delta
                eval_config["quad_errors"][qe_ix]["delta"] = quadrupole_error_delta
                if verbose:
                    print("_run_evaluation()/ ", qe_ix, qe)
            else:
                eval_config["quad_errors"][qe_ix]["delta"] = 0.0

    # Apply quad_tilt_errors
    if eval_config["quad_tilt_errors"]:
        for qe_ix, qe in enumerate(eval_config["quad_tilt_errors"]):
            if (
                eval_config["quad_tilt_errors"][qe_ix]["FODO_index"]
                in fodo_indices_with_quad_tilt_error
            ):
                quadrupole_tilt_error_delta = sampling_func(
                    mean_or_min_quad_tilt_error, std_or_max_quad_tilt_error
                )
                print("quadrupole_tilt_error_delta: ", quadrupole_tilt_error_delta)

                # TODO(aribra): # Target prediction (Quadrupole Tilt)
                # quadrupole_errors_target_values[qe_ix] = quadrupole_tilt_error_delta

                eval_config["quad_tilt_errors"][qe_ix]["tilt_angle"] = (
                    quadrupole_tilt_error_delta
                )
                if verbose:
                    print("_run_evaluation()/ ", qe_ix, qe)
            else:
                eval_config["quad_tilt_errors"][qe_ix]["tilt_angle"] = 0.0

    # Apply dipole_tilt_errors
    if eval_config["dipole_tilt_errors"]:
        for qe_ix, qe in enumerate(eval_config["dipole_tilt_errors"]):
            if (
                eval_config["dipole_tilt_errors"][qe_ix]["FODO_index"]
                in fodo_indices_with_dipole_tilt_error
            ):
                dipole_tilt_error_delta = sampling_func(
                    mean_or_min_dipole_tilt_error, std_or_max_dipole_tilt_error
                )

                # TODO(aribra): # Target prediction (Dipole Tilt)
                # quadrupole_errors_target_values[qe_ix] = dipole_tilt_error_delta

                eval_config["dipole_tilt_errors"][qe_ix]["tilt_angle"] = (
                    dipole_tilt_error_delta
                )
                if verbose:
                    print("_run_evaluation()/ ", qe_ix, qe)
            else:
                eval_config["dipole_tilt_errors"][qe_ix]["tilt_angle"] = 0.0

    if verbose:
        print("evaluate_model()/ base_configurations: ", eval_config)

    # * Simulate without error (baseline) + after applying the error
    sim_runner = SimulationRunner(
        base_configurations=[eval_config], common_parameters=common_parameters
    )

    initial_states = None

    sim_runner.run_configurations(
        draw_plots=False, verbose=verbose, initial_states=initial_states, run_no_error_sim=False
    )

    initial_states = sim_runner.initial_states

    simulator_no_error = sim_runner.simulators_no_error.get(
        f"{eval_config['config_name']} - No Error"
    )
    simulator_with_error = sim_runner.simulators_with_error.get(
        f"{eval_config['config_name']} - With Error"
    )

    # * The initial_states are the same in both simulations

    merged_config = {**common_parameters, **eval_config}

    # BPM Random Noise handling (for include_bpm_noise additional error source)
    if include_bpm_noise and bpm_noise_value > 0:
        if verbose:
            print(f"Applying random BPM noise ±{bpm_noise_value}.")
        simulator_with_error.bpm_readings["x"] = simulator_with_error.bpm_readings[
            "x"
        ] + np.random.uniform(
            -bpm_noise_value,
            bpm_noise_value,
            simulator_with_error.bpm_readings["x"].shape,
        )
        simulator_with_error.bpm_readings["y"] = simulator_with_error.bpm_readings[
            "y"
        ] + np.random.uniform(
            -bpm_noise_value,
            bpm_noise_value,
            simulator_with_error.bpm_readings["y"].shape,
        )

    # Original BPM noise handling for noise_type == 'bpm'
    if noise_type == "bpm" and noise_level > 0:
        if verbose:
            print(f"Applying random BPM noise to X and Y axis ±{noise_level}.")
        simulator_with_error.bpm_readings["x"] = simulator_with_error.bpm_readings[
            "x"
        ] + np.random.uniform(
            -noise_level, noise_level, simulator_with_error.bpm_readings["x"].shape
        )
        simulator_with_error.bpm_readings["y"] = simulator_with_error.bpm_readings[
            "y"
        ] + np.random.uniform(
            -noise_level, noise_level, simulator_with_error.bpm_readings["y"].shape
        )

    # # BPM Systematic Shift handling
    # if noise_type == 'bpm_shift' and noise_level != 0:
    #     shift_axes = []
    #     if x_shift:
    #         shift_axes.append('x')
    #     if y_shift:
    #         shift_axes.append('y')

    #     if verbose:
    #         print(f"Applying systematic shift of {noise_level*1e6:.1f}μm to BPM readings on axes: {shift_axes}")

    #     for axis in shift_axes:
    #         if axis in simulator_with_error.bpm_readings:
    #             simulator_with_error.bpm_readings[axis] = simulator_with_error.bpm_readings[axis] + noise_level
    #             if verbose:
    #                 print(f"  Applied {noise_level*1e6:.1f}μm shift to {axis}-axis")

    # Create SimulationDataset instance
    simulation_dataset = SimulationDataset(
        merged_config=merged_config,
        bpm_readings_no_error=simulator_no_error.bpm_readings,
        bpm_readings_with_error=simulator_with_error.bpm_readings,
        bpm_positions=simulator_no_error.bpm_positions,
        quadrupole_errors=simulator_with_error.quad_errors,
        quadrupole_tilt_errors=simulator_with_error.quadrupole_tilt_errors,
        dipole_tilt_errors=simulator_with_error.dipole_tilt_errors,
        apply_avg=common_parameters.get("apply_avg", False),
    )

    # Generate data using the same parameters as during training
    start_rev = common_parameters.get("start_rev", 0)
    end_rev = common_parameters.get("end_rev", simulator_no_error.n_turns)
    fodo_cell_indices = common_parameters.get(
        "fodo_cell_indices", list(range(simulator_no_error.n_FODO))
    )
    planes = common_parameters.get("planes", ["x", "y"])

    if verbose:
        print(
            f"[Evaluation - Generate data params:]\n"
            f"\t start_rev={start_rev}\n"
            f"\t end_rev={end_rev}\n"
            f"\t fodo_cell_indices={fodo_cell_indices}\n"
            f"\t planes={planes}"
        )

    (
        input_tensor,
        target_tensor,
        error_values_quad_misalign,
        error_values_quad_tilt,
        error_values_dipole_tilt,
    ) = simulation_dataset.process_simulated_data(
        start_rev, end_rev, fodo_cell_indices, planes
    )

    print("-------------------target_tensor: ", target_tensor)

    torch.save(input_tensor, "notebooks/input_tensor_eval.pt")

    # Reshape input data to match model input
    n_samples, n_turns, n_BPMs, n_planes = input_tensor.shape
    input_size = n_BPMs * n_planes
    input_data = input_tensor.reshape(n_samples, n_turns, input_size)

    # **Reshape to (-1, n_planes) for scaling**
    input_data_flat = input_data.reshape(
        n_samples * n_turns * n_BPMs, n_planes
    )  # Shape: (n_samples * n_turns * n_BPMs, n_planes)

    # **Use the input scaler to transform the input data**
    input_data_flat_scaled = dataset_scalers["input_scaler"].transform(input_data_flat)

    # **Reshape back to (n_samples, n_turns, n_BPMs, n_planes)**
    input_data_scaled = input_data_flat_scaled.reshape(
        n_samples, n_turns, n_BPMs, n_planes
    )

    # **Flatten to (n_samples, n_turns, input_size) for model input**
    input_data_scaled = input_data_scaled.reshape(n_samples, n_turns, input_size)

    # Convert to tensor
    input_tensor_model = torch.tensor(input_data_scaled, dtype=torch.float32).to(device)

    # Predict the error
    model.eval()
    with torch.no_grad():
        if verbose:
            print("input to model: ", input_tensor_model.shape)
        predicted_error = model(
            input_tensor_model
        )  # predicted_error shape: (n_samples, output_size)
        if verbose:
            print("predicted_errors = ", predicted_error)
        predicted_errors_scaled_values = predicted_error.cpu().numpy()

    # Inverse transform the prediction
    predicted_errors_values_transformed_back = dataset_scalers[
        "target_scaler"
    ].inverse_transform(predicted_errors_scaled_values)
    predicted_errors_values_transformed_back = (
        predicted_errors_values_transformed_back.flatten()
    )
    if verbose:
        print(
            f"predicted_errors_values_transformed_back = {predicted_errors_values_transformed_back}"
        )

    actual_deltas = {}
    predicted_deltas = {}

    for pesv_ix, pesv in enumerate(predicted_errors_values_transformed_back):
        predicted_error_value = pesv
        predicted_deltas[pesv_ix] = predicted_error_value
        if pesv_ix not in quadrupole_errors_target_values:
            if verbose:
                print(
                    f"WARNING - error prediction output with index={pesv_ix}, is not available.\n\tthis may indicate that you set custom error config rather than the trained network"
                )
            continue
        actual_deltas[pesv_ix] = quadrupole_errors_target_values[pesv_ix]
        if verbose:
            print(
                f"\tActual quadrupole error delta: {quadrupole_errors_target_values[pesv_ix]:.7e}, {quadrupole_errors_target_values[pesv_ix] * 1e6}"
            )
            print(
                f"\tPredicted quadrupole error delta: {predicted_error_value:.7e}, {predicted_error_value * 1e6}"
            )
            print("---")

    # * Apply correction and re-run simulation
    # Correct the quadrupole error by subtracting the predicted error
    corrected_deltas = {}
    for pevtb_ix, pevtb in predicted_deltas.items():
        if pevtb_ix not in quadrupole_errors_target_values:
            if verbose:
                print(
                    f"WARNING - error prediction output with index={pevtb_ix}, is not available.\n\tthis may indicate that you set custom error config rather than the trained network"
                )
            # If error was not found, we assume it is 0
            corrected_deltas[pevtb_ix] = 0.0
            continue
        corrected_delta = quadrupole_errors_target_values[pevtb_ix] - pevtb
        corrected_deltas[pevtb_ix] = corrected_delta
        if verbose:
            print(
                f"\tCorrected_delta quadrupole error delta [0]: {corrected_delta:.7e}, {corrected_delta * 1e6}"
            )

    if merged_config["target_data"] == "quad_misalign_deltas":
        target_errors_key = "quad_errors"
    elif merged_config["target_data"] == "quad_tilt_angles":
        target_errors_key = "quad_tilt_errors"
    elif merged_config["target_data"] == "dipole_tilt_angles":
        target_errors_key = "dipole_tilt_errors"

    eval_config_corrected = eval_config.copy()
    for cord_ix, cord in corrected_deltas.items():
        if cord_ix < len(eval_config_corrected[target_errors_key]):
            eval_config_corrected[target_errors_key][cord_ix]["delta"] = cord
            if verbose:
                print(cord_ix, cord)

    runner_corrected = SimulationRunner(
        base_configurations=[eval_config_corrected], common_parameters=common_parameters
    )

    runner_corrected.run_configurations(
        draw_plots=False, verbose=False, initial_states=initial_states
    )
    simulator_corrected = runner_corrected.simulators_with_error.get(
        f"{eval_config_corrected['config_name']} - With Error"
    )

    # * Compare y positions after applying the correction with the original simulation without errors
    # Use the same start_rev and end_rev

    if plot:
        # Extract BPM readings for comparison
        bpm_readings_no_error = simulator_no_error.bpm_readings["y"][
            :, start_rev:end_rev, :
        ][:, :, fodo_cell_indices].mean(axis=0)  # Shape: [n_turns, n_BPMs]
        bpm_readings_with_error = simulator_with_error.bpm_readings["y"][
            :, start_rev:end_rev, :
        ][:, :, fodo_cell_indices].mean(axis=0)  # Shape: [n_turns, n_BPMs]
        bpm_readings_corrected = simulator_corrected.bpm_readings["y"][
            :, start_rev:end_rev, :
        ][:, :, fodo_cell_indices].mean(axis=0)  # Shape: [n_turns, n_BPMs]

        bpm_indx = 3
        rev_numbers = np.arange(end_rev - 100, end_rev)
        if verbose:
            print(f"rev_numbers = {rev_numbers}")
            print(f"bpm_readings_no_error = {bpm_readings_no_error.shape}")
        # Plot the comparison
        plt.figure(figsize=(12, 6))
        plt.plot(
            rev_numbers,
            bpm_readings_no_error[-100:, bpm_indx],
            "-o",
            label="No Error",
            color="blue",
        )
        plt.plot(
            rev_numbers,
            bpm_readings_with_error[-100:, bpm_indx],
            "-x",
            label="With Error",
            color="red",
        )
        plt.plot(
            rev_numbers,
            bpm_readings_corrected[-100:, bpm_indx],
            "-v",
            label="After Correction",
            color="green",
        )
        plt.xlabel("Turn")
        plt.ylabel(f"Average y position at BPM {bpm_indx}")
        plt.title("Comparison of y positions after correction")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Compute and print the residual error after correction
        residual_error = np.abs(bpm_readings_no_error - bpm_readings_corrected).mean()
        print(
            f"Residual error after correction (mean absolute difference): {residual_error:.6e}, {residual_error * 1e6}"
        )

        # Additional plots
        simulator_no_error.plot_comparison(
            simulator_with_error,
            cell_idx=bpm_indx,
            viz_start_idx=end_rev - 100,
            viz_end_idx=end_rev,
            save_label="WOEvsWE",
            window_size=50,
            plot_all=True,
            extra_title="Before correction",
        )

        simulator_no_error.plot_comparison(
            simulator_corrected,
            cell_idx=bpm_indx,
            viz_start_idx=end_rev - 100,
            viz_end_idx=end_rev,
            save_label="WOEvsC",
            window_size=50,
            plot_all=True,
            extra_title="After correction",
        )

        simulator_no_error.plot_bpm_heatmaps(
            cell_idx=bpm_indx, simulation_label="No Error"
        )
        simulator_with_error.plot_bpm_heatmaps(
            cell_idx=bpm_indx, simulation_label="With Error"
        )
        simulator_corrected.plot_bpm_heatmaps(
            cell_idx=bpm_indx, simulation_label="Corrected"
        )

        simulator_corrected.plot_bpm_comparison_last_images(
            simulator_corrected,
            simulator_with_error,
            cell_idx=bpm_indx,
            save_label="WEvsC",
            particles="all",
        )
        simulator_corrected.plot_bpm_comparison_last_images(
            simulator_corrected,
            simulator_no_error,
            cell_idx=bpm_indx,
            save_label="WOEvsC",
            particles="all",
        )

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
        verbose=True,
    )


def benchmark_evaluation_bpm_noise(
    model,
    base_configurations,
    common_parameters,
    dataset_scalers,
    noise_start=0,
    noise_stop=100e-6,
    bins=11,
    runs=20,
    k_errors_config=None,
    include_quad_tilt=False,
    quad_tilt_range=(0.01, 0.05),
):
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
    - k_errors_config: K errors configuration dict.
    - include_quad_tilt: Include quad tilt as additional error source.
    - quad_tilt_range: Quad tilt range if include_quad_tilt is True.

    Returns:
    - stats: Dictionary containing statistics for each noise level and FODO index.
    """

    noise_pallette = np.linspace(
        noise_start, noise_stop, bins
    )  # Convert to meters if needed

    # Initialize a dictionary to store statistics
    stats = defaultdict(
        lambda: defaultdict(list)
    )  # stats[noise_level][fodo_index] = list of errors

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
                noise_type="bpm",
                noise_level=noise_level,
                plot=False,
                verbose=True,
                k_errors_config=k_errors_config,
                include_quad_tilt=include_quad_tilt,
                quad_tilt_range=quad_tilt_range,
            )
            for fodo_ix in actual_deltas:
                error = np.abs(actual_deltas[fodo_ix] - predicted_deltas[fodo_ix])
                stats[noise_level][fodo_ix].append(error)
        print(f"Completed benchmarking for noise_level={noise_level} meters.")

    return stats


def benchmark_evaluation_tilt_noise(
    model,
    base_configurations,
    common_parameters,
    dataset_scalers,
    noise_start=10,
    noise_stop=50,
    bins=5,
    runs=50,
    k_errors_config=None,
    include_bpm_noise=False,
    bpm_noise_range=(0, 100e-6),
):
    """
    Automates running the evaluation with varying levels of Quad Tilt noise.
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
    - k_errors_config: K errors configuration dict.
    - include_bpm_noise: Include BPM noise as additional error source.
    - bpm_noise_range: BPM noise range if include_bpm_noise is True.

    Returns:
    - stats: Dictionary containing statistics for each noise level and FODO index.
    """

    noise_pallette = np.linspace(noise_start, noise_stop, bins)

    # Initialize a dictionary to store statistics
    stats = defaultdict(
        lambda: defaultdict(list)
    )  # stats[noise_level][fodo_index] = list of errors

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
                noise_type="quad_tilt",
                noise_level=noise_level,
                plot=False,
                verbose=False,
                k_errors_config=k_errors_config,
                include_bpm_noise=include_bpm_noise,
                bpm_noise_range=bpm_noise_range,
            )
            for fodo_ix in actual_deltas:
                error = np.abs(actual_deltas[fodo_ix] - predicted_deltas[fodo_ix])
                stats[noise_level][fodo_ix].append(error)
        print(f"Completed benchmarking for noise_level={noise_level} meters.")

    return stats


def benchmark_evaluation_k_errors(
    model,
    base_configurations,
    common_parameters,
    dataset_scalers,
    k_errors_config=None,
    noise_start=0.04,
    noise_stop=0.04,
    bins=11,
    runs=50,
    include_quad_tilt=False,
    quad_tilt_range=(0.01, 0.05),
    include_bpm_noise=False,
    bpm_noise_range=(0, 100e-6),
):
    """
    Automates running the evaluation with varying levels of K errors (systemic drift).
    Collects statistics and plots model prediction accuracy against k_drift levels.

    Parameters:
    - model: The trained model.
    - base_configurations: Base configurations for the simulation.
    - common_parameters: Common parameters for the simulation.
    - dataset_scalers: Dictionary containing 'input_scaler' and 'target_scaler'.
    - k_errors_config: K errors configuration dict.
    - noise_start: Minimum k_drift fraction to apply.
    - noise_stop: Maximum k_drift fraction to apply.
    - bins: Number of bins for k_drift levels.
    - runs: Number of evaluations to run per k_drift level.
    - include_quad_tilt: Include quad tilt as additional error source.
    - quad_tilt_range: Quad tilt range if include_quad_tilt is True.
    - include_bpm_noise: Include BPM noise as additional error source.
    - bpm_noise_range: BPM noise range if include_bpm_noise is True.

    Returns:
    - stats: Dictionary containing statistics for each k_drift level and FODO index.
    """

    noise_pallette = np.linspace(noise_start, noise_stop, bins)

    # Initialize a dictionary to store statistics
    stats = defaultdict(
        lambda: defaultdict(list)
    )  # stats[noise_level][fodo_index] = list of errors

    for noise_level in noise_pallette:
        print("Running evaluation for k_drift=", noise_level)
        if model.training:
            model.eval()  # Ensure model is in evaluation mode

        # Create modified k_errors_config with the current noise_level
        if k_errors_config is not None:
            current_k_errors = k_errors_config.copy()
            current_k_errors["k_systemic_drift_fraction_range"] = (
                noise_level,
                noise_level,
            )
        else:
            current_k_errors = {
                "enabled": True,
                "k_systemic_drift_fraction_range": (noise_level, noise_level),
                "k_stochastic_jitter_fraction_range": (0.005, 0.01),
                "k_error_cells": [
                    {"FODO_index": i, "quad_type": "focusing"} for i in range(8)
                ]
                + [{"FODO_index": i, "quad_type": "defocusing"} for i in range(8)],
            }

        for run in range(runs):
            print(f"\t-------------[Run {run + 1}/{runs}]")
            actual_deltas, predicted_deltas = _run_evaluation(
                model=model,
                base_configurations=base_configurations,
                common_parameters=common_parameters,
                dataset_scalers=dataset_scalers,
                noise_type="k_errors",
                noise_level=noise_level,
                plot=False,
                verbose=False,
                k_errors_config=current_k_errors,
                include_quad_tilt=include_quad_tilt,
                quad_tilt_range=quad_tilt_range,
                include_bpm_noise=include_bpm_noise,
                bpm_noise_range=bpm_noise_range,
            )
            for fodo_ix in actual_deltas:
                error = np.abs(actual_deltas[fodo_ix] - predicted_deltas[fodo_ix])
                stats[noise_level][fodo_ix].append(error)
        print(f"Completed benchmarking for k_drift={noise_level}")

    return stats


def split_merged_config(merged_config):
    # Keys that belong to base_configurations
    base_keys = {
        "config_name",
        "design_radius",
        "n_FODO",
        "f",
        "L_quad",
        "L_straight",
        "quad_errors",
        "quad_tilt_errors",
        "dipole_tilt_errors",
        "total_dipole_bending_angle",
    }

    # Extract base configuration parameters
    base_config = {k: v for k, v in merged_config.items() if k in base_keys}

    # Wrap the base_config in a list as
    base_configurations = [base_config]

    # Extract common parameters by excluding base_keys
    common_parameters = {k: v for k, v in merged_config.items() if k not in base_keys}

    return base_configurations, common_parameters


def main_evaluation_block(
    model,
    data_sub_cfg,
    val_loader=None,
    primary_benchmark=None,
    run_benchmark=False,
    bins=11,
    runs=50,
    # K errors
    enable_k_errors=False,
    k_drift_range=(0.04, 0.04),
    k_jitter_range=(0.005, 0.01),
    # Quad tilt
    include_quad_tilt=False,
    quad_tilt_range=(0.01, 0.05),
    # BPM noise
    include_bpm_noise=False,
    bpm_noise_range=(0, 100e-6),
    # BPM shift
    shift_range=(-100e-6, 100e-6),
    x_shift=False,
    y_shift=False,
    # Data
    data_dir=None,
):
    """
    Main evaluation function to run model evaluation or benchmarking.

    Args:
        model: The trained model.
        data_sub_cfg: Dictionary containing configuration data (merged_config, scalers, etc.).
        val_loader: DataLoader for validation data.
        primary_benchmark: Type of primary benchmark ('k_errors', 'bpm_noise', 'quad_tilt', 'bpm_shift').
        run_benchmark: Boolean to indicate whether to run benchmarking or single evaluation.
        bins: Number of bins for noise/shift levels in benchmarks.
        runs: Number of runs per noise/shift level in benchmarks.
        enable_k_errors: Enable k_errors as error source.
        k_drift_range: K systemic drift fraction range.
        k_jitter_range: K stochastic jitter fraction range.
        include_quad_tilt: Include quad_tilt as additional error source.
        quad_tilt_range: Quad tilt angle range in mrads.
        include_bpm_noise: Include BPM noise as additional error source.
        bpm_noise_range: BPM noise range in meters.
        shift_range: BPM shift range in meters for bpm_shift benchmark.
        x_shift: Apply shifts on X-axis for bpm_shift benchmark.
        y_shift: Apply shifts on Y-axis for bpm_shift benchmark.
        data_dir: Data directory for simulation-based benchmarks.
    """

    merged_config = data_sub_cfg["merged_config"]

    print("\n." * 10)
    print(merged_config)
    print("\n." * 10)

    input_scaler_config = data_sub_cfg["input_scaler_config"]
    target_scaler_config = data_sub_cfg["target_scaler_config"]
    overridden_base_config = data_sub_cfg["overridden_base_config"]

    # Data scalers
    dataset_scalers = {
        "input_scaler": deserialize_minmax_scaler(input_scaler_config),
        "target_scaler": deserialize_minmax_scaler(target_scaler_config),
    }

    # Flags for evaluation mode
    run_evaluate_once = True
    if run_benchmark and primary_benchmark is not None:
        run_evaluate_once = False

    # Build k_errors dict if enabled
    k_errors_config = None
    if enable_k_errors:
        k_errors_config = {
            "enabled": True,
            "k_systemic_drift_fraction_range": k_drift_range,
            "k_stochastic_jitter_fraction_range": k_jitter_range,
            "k_error_cells": [
                {"FODO_index": i, "quad_type": "focusing"} for i in range(8)
            ]
            + [{"FODO_index": i, "quad_type": "defocusing"} for i in range(8)],
        }
        print(
            f"K Errors enabled: drift_range={k_drift_range}, jitter_range={k_jitter_range}"
        )

    # Initialize benchmark parameters based on primary_benchmark
    NOISE_START, NOISE_STOP = None, None
    NOISE_PALLETTE = None

    if primary_benchmark == "k_errors":
        print("Running benchmark for K errors...")
        NOISE_START, NOISE_STOP = k_drift_range
        NOISE_PALLETTE = np.linspace(NOISE_START, NOISE_STOP, bins)
    elif primary_benchmark == "bpm_noise":
        print("Running benchmark for BPM noise...")
        NOISE_START, NOISE_STOP = bpm_noise_range
        NOISE_PALLETTE = np.linspace(NOISE_START, NOISE_STOP, bins)
    elif primary_benchmark == "quad_tilt":
        print("Running benchmark for Quadrupole Tilt noise...")
        NOISE_START, NOISE_STOP = quad_tilt_range
        NOISE_PALLETTE = np.linspace(NOISE_START, NOISE_STOP, bins)
    elif primary_benchmark == "bpm_shift":
        print("Running benchmark for BPM shift...")
        NOISE_START, NOISE_STOP = shift_range
        NOISE_PALLETTE = np.linspace(NOISE_START, NOISE_STOP, bins)
        print(
            f"BPM Shift Benchmark - Testing axes: {'x' if x_shift else ''}{'y' if y_shift else ''}"
        )

    CANCEL_TILT_ERROR = False
    CANCEL_MISALIGN_ERROR = False

    # Split merged_config into base_configurations and common_parameters
    base_configurations, common_parameters = split_merged_config(merged_config)

    # Merge overridden_base_config into base_configurations[0] to preserve list structure
    if isinstance(overridden_base_config, dict):
        base_configurations[0].update(overridden_base_config)
    elif isinstance(overridden_base_config, list):
        base_configurations = overridden_base_config  # Use directly if already a list

    # Set up FODO mapping dictionary
    fodo_mapping = {}
    if merged_config["target_data"] == "quad_misalign_deltas":
        target_errors_cfg = merged_config["quad_errors"]
    elif merged_config["target_data"] == "quad_tilt_angles":
        target_errors_cfg = merged_config["quad_tilt_errors"]
    elif merged_config["target_data"] == "dipole_tilt_angles":
        target_errors_cfg = merged_config["dipole_tilt_errors"]

    for qe_ix, qe in enumerate(target_errors_cfg):
        fodo_mapping[qe_ix] = qe["FODO_index"]

    benchmark_info = {
        "benchmark_type": primary_benchmark,
        "noise_start": NOISE_START,
        "noise_stop": NOISE_STOP,
        "bins": bins,
        "noise_pallette": NOISE_PALLETTE,
        "runs_per_noise": runs,
        "fodo_mapping": fodo_mapping,
        "cancel_tilt_error": CANCEL_TILT_ERROR,
        "cancel_misalign_error": CANCEL_MISALIGN_ERROR,
        "x_shift": x_shift,
        "y_shift": y_shift,
        "enable_k_errors": enable_k_errors,
        "k_drift_range": k_drift_range,
        "k_jitter_range": k_jitter_range,
        "include_quad_tilt": include_quad_tilt,
        "include_bpm_noise": include_bpm_noise,
    }

    if CANCEL_TILT_ERROR:
        base_configurations[0]["quad_tilt_errors"] = []
        base_configurations[0]["dipole_tilt_errors"] = []

    if CANCEL_MISALIGN_ERROR:
        base_configurations[0]["quad_errors"] = []

    if CANCEL_TILT_ERROR and CANCEL_MISALIGN_ERROR:
        common_parameters["target_data"] = False

    # Insert k_errors into base_configurations if enabled
    if k_errors_config is not None:
        base_configurations[0]["k_errors"] = k_errors_config

    # common_parameters['num_particles'] = 10

    if run_evaluate_once:
        evaluate_once(model, base_configurations, common_parameters, dataset_scalers)

    elif run_benchmark:
        if CANCEL_MISALIGN_ERROR and CANCEL_TILT_ERROR:
            print("BENCHMARK was not run!!")
            return

        # For bpm_shift, use existing data
        if primary_benchmark == "bpm_shift":
            print("Running bpm_shift benchmark with existing data...")
            stats = benchmark_evaluation_bpm_shift(
                model=model,
                base_configurations=base_configurations,
                common_parameters=common_parameters,
                val_loader=val_loader,
                dataset_scalers=dataset_scalers,
                merged_config=merged_config,
                shift_start=NOISE_START,
                shift_stop=NOISE_STOP,
                bins=bins,
                runs=runs,
                x_shift=x_shift,
                y_shift=y_shift,
            )

            stats["benchmark_info"] = benchmark_info
            save_stats_path = f"{SAVE_DIR_BENCHMARKS}/benchmark_stats_bpm_shift_MisAlign-True_Tilt-True.pt"
            print(f"save_stats_path: {save_stats_path}")
            torch.save(convert_defaultdict_to_dict(stats), save_stats_path)
            plot_benchmark_stats(stats, benchmark_info)

        elif primary_benchmark == "k_errors":
            print("Running k_errors benchmark...")
            stats = benchmark_evaluation_k_errors(
                model=model,
                base_configurations=base_configurations,
                common_parameters=common_parameters,
                dataset_scalers=dataset_scalers,
                k_errors_config=k_errors_config,
                noise_start=NOISE_START,
                noise_stop=NOISE_STOP,
                bins=bins,
                runs=runs,
                include_quad_tilt=include_quad_tilt,
                quad_tilt_range=quad_tilt_range,
                include_bpm_noise=include_bpm_noise,
                bpm_noise_range=bpm_noise_range,
            )

            stats["benchmark_info"] = benchmark_info
            save_stats_path = f"{SAVE_DIR_BENCHMARKS}/benchmark_stats_k_errors.pt"
            print(f"save_stats_path: {save_stats_path}")
            torch.save(convert_defaultdict_to_dict(stats), save_stats_path)
            plot_benchmark_stats(stats, benchmark_info)

        else:
            if primary_benchmark == "bpm_noise":
                print("Running bpm_noise benchmark...")
                stats = benchmark_evaluation_bpm_noise(
                    model=model,
                    base_configurations=base_configurations,
                    common_parameters=common_parameters,
                    dataset_scalers=dataset_scalers,
                    noise_start=NOISE_START,
                    noise_stop=NOISE_STOP,
                    bins=bins,
                    runs=runs,
                    k_errors_config=k_errors_config,
                    include_quad_tilt=include_quad_tilt,
                    quad_tilt_range=quad_tilt_range,
                )

                stats["benchmark_info"] = benchmark_info
                save_stats_path = f"{SAVE_DIR_BENCHMARKS}/benchmark_stats_bpm_MisAlign-True_Tilt_True.pt"
                print(f"save_stats_path: {save_stats_path}")
                torch.save(convert_defaultdict_to_dict(stats), save_stats_path)
                plot_benchmark_stats(stats, benchmark_info)

            elif primary_benchmark == "quad_tilt":
                print("Running quad_tilt benchmark...")
                stats = benchmark_evaluation_tilt_noise(
                    model=model,
                    base_configurations=base_configurations,
                    common_parameters=common_parameters,
                    dataset_scalers=dataset_scalers,
                    noise_start=NOISE_START,
                    noise_stop=NOISE_STOP,
                    bins=bins,
                    runs=runs,
                    k_errors_config=k_errors_config,
                    include_bpm_noise=include_bpm_noise,
                    bpm_noise_range=bpm_noise_range,
                )

                stats["benchmark_info"] = benchmark_info
                save_stats_path = f"{SAVE_DIR_BENCHMARKS}/benchmark_stats_quad_tilt_MisAlign-True_Tilt-True.pt"
                print(f"save_stats_path: {save_stats_path}")
                torch.save(convert_defaultdict_to_dict(stats), save_stats_path)
                plot_benchmark_stats(stats, benchmark_info)


def benchmark_evaluation_bpm_shift(
    model,
    base_configurations,
    common_parameters,
    val_loader=None,
    dataset_scalers=None,
    merged_config=None,
    shift_start=0,
    shift_stop=100e-6,
    bins=11,
    runs=20,
    x_shift=True,
    y_shift=True,
):
    """
    Tests model reliability with systematic BPM shifts on X/Y axes using either existing data or simulated data.

    Parameters:
    - model: The trained model.
    - val_loader: DataLoader for existing validation dataset (optional).
    - dataset_scalers: Dictionary containing 'input_scaler' and 'target_scaler' (required if val_loader is provided).
    - merged_config: Merged configuration dictionary (required if val_loader is provided).
    - shift_start: Minimum shift level to apply (in meters).
    - shift_stop: Maximum shift level to apply (in meters).
    - bins: Number of shift levels to test.
    - runs: Number of evaluations to run per shift level.
    - x_shift: Whether to apply shifts on X-axis.
    - y_shift: Whether to apply shifts on Y-axis.

    Returns:
    - stats: Dictionary containing statistics for each shift level and FODO index.
    """
    if val_loader is not None and (dataset_scalers is None or merged_config is None):
        raise ValueError(
            "dataset_scalers and merged_config must be provided when using val_loader"
        )

    shift_levels = np.linspace(shift_start, shift_stop, bins)

    # Initialize a dictionary to store statistics
    stats = defaultdict(
        lambda: defaultdict(list)
    )  # stats[shift_level][fodo_index] = list of errors

    shift_axes = []
    if x_shift:
        shift_axes.append("x")
    if y_shift:
        shift_axes.append("y")

    if not shift_axes:
        raise ValueError("At least one of x_shift or y_shift must be True")

    print(f"BPM Shift Benchmark - Testing axes: {shift_axes}")

    print("val_loader = ", val_loader)

    if val_loader is not None:
        for shift_level in shift_levels:
            print(f"Testing BPM shift: {shift_level * 1e6:.1f}μm on axes {shift_axes}")
            if model.training:
                model.eval()

            for run in range(runs):
                print(f"\t-------------[Run {run + 1}/{runs}]")
                actual_deltas, predicted_deltas = evaluate_with_existing_data(
                    model=model,
                    val_loader=val_loader,
                    dataset_scalers=dataset_scalers,
                    merged_config=merged_config,
                    noise_type="bpm_shift",
                    noise_level=shift_level,
                    x_shift=x_shift,
                    y_shift=y_shift,
                    verbose=False,
                )
                for fodo_ix in actual_deltas:
                    error = np.abs(actual_deltas[fodo_ix] - predicted_deltas[fodo_ix])
                    stats[shift_level][fodo_ix].append(error)

                print(
                    f"Completed benchmarking for shift_level={shift_level * 1e6:.1f}μm."
                )
    else:
        print("val_loader is None 00000000000000000000000000000000000000000000")

        for shift_level in shift_levels:
            print(f"Testing BPM shift: {shift_level * 1e6:.1f}μm on axes {shift_axes}")
            if model.training:
                model.eval()

            for run in range(runs):
                print(f"\t-------------[Run {run + 1}/{runs}]")
                actual_deltas, predicted_deltas = _run_evaluation(
                    model=model,
                    base_configurations=base_configurations,
                    common_parameters=common_parameters,
                    dataset_scalers=dataset_scalers,
                    noise_type="bpm_shift",
                    noise_level=shift_level,
                    x_shift=x_shift,
                    y_shift=y_shift,
                    plot=False,
                    verbose=False,
                )
                for fodo_ix in actual_deltas:
                    error = np.abs(actual_deltas[fodo_ix] - predicted_deltas[fodo_ix])
                    stats[shift_level][fodo_ix].append(error)

                print(
                    f"Completed benchmarking for shift_level={shift_level * 1e6:.1f}μm."
                )

    return stats
