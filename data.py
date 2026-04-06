# data.py

import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from sim_config import base_configurations, common_parameters
from automate_dataset_collection import DataAutomation
from constants import Constants as C
from utils import serialize_minmax_scaler
from sim_config import SAVE_DIR_FIGS


def gen_data(n_simulations=0):
    """Data Generation using DataAutomation
    Data is saved automatically. The method DataAutomation.save_data() is called after each simualtion
    to save accumulated data while runing. This prevents losing all the data when running large number of
    simualtions.

    Args:
        n_simulations (int, optional): Defaults to 0.
    """

    # Create an instance of DataAutomation with delta_range
    data_automation = DataAutomation(
        base_configurations, common_parameters, n_simulations
    )
    dataset_scalers = data_automation.get_data_scalers()

    data_tag = data_automation.tag
    print("Using tag:", data_tag)

    # Run the data automation process and get the data tensors
    data_automation.run(
        include_no_error_data=common_parameters["include_no_error_data"],
        skip_data_on_delta_ranges=False,
    )

    data_dir = os.path.join(f"data/{data_tag}")
    return load_data_from_dir(data_dir=data_dir)


def load_data_from_dir(data_dir=None, override_config=False):
    """Load data from a directory"""
    if not os.path.exists(data_dir):
        raise FileNotFoundError("Data directory does not exist.")

    if data_dir:
        postfix = "final"

        sim_data = {}

        sim_data[C.DATA_KEY_ALL_ERROR_VALUES_DIPOLE_TILT] = torch.load(
            f"{data_dir}/{C.DATA_KEY_ALL_ERROR_VALUES_DIPOLE_TILT}-{postfix}.pt",
            weights_only=True,
        )
        sim_data[C.DATA_KEY_ALL_ERROR_VALUES_QUAD_MISALIGN] = torch.load(
            f"{data_dir}/{C.DATA_KEY_ALL_ERROR_VALUES_QUAD_MISALIGN}-{postfix}.pt",
            weights_only=True,
        )
        sim_data[C.DATA_KEY_ALL_ERROR_VALUES_QUAD_TILT] = torch.load(
            f"{data_dir}/{C.DATA_KEY_ALL_ERROR_VALUES_QUAD_TILT}-{postfix}.pt",
            weights_only=True,
        )
        sim_data[C.DATA_KEY_DATA_AUTOMATION] = torch.load(
            f"{data_dir}/{C.DATA_KEY_DATA_AUTOMATION}-{postfix}.pt"
        )
        sim_data[C.DATA_KEY_DATASET_SCALERS] = torch.load(
            f"{data_dir}/{C.DATA_KEY_DATASET_SCALERS}-{postfix}.pt"
        )
        sim_data[C.DATA_KEY_INPUT_TENSORS] = torch.load(
            f"{data_dir}/{C.DATA_KEY_INPUT_TENSORS}-{postfix}.pt", weights_only=True
        )
        sim_data[C.DATA_KEY_INPUT_TENSORS_SCALED] = torch.load(
            f"{data_dir}/{C.DATA_KEY_INPUT_TENSORS_SCALED}-{postfix}.pt",
            weights_only=True,
        )
        sim_data[C.DATA_KEY_MERGED_CONFIG] = torch.load(
            f"{data_dir}/{C.DATA_KEY_MERGED_CONFIG}-{postfix}.pt"
        )
        sim_data[C.DATA_KEY_TARGET_TENSORS] = torch.load(
            f"{data_dir}/{C.DATA_KEY_TARGET_TENSORS}-{postfix}.pt", weights_only=True
        )
        sim_data[C.DATA_KEY_TARGET_TENSORS_SCALED] = torch.load(
            f"{data_dir}/{C.DATA_KEY_TARGET_TENSORS_SCALED}-{postfix}.pt",
            weights_only=True,
        )

        data_automation = sim_data[C.DATA_KEY_DATA_AUTOMATION]

        base_configuration = [data_automation.overridden_base_config]
        input_tensors_scaled = sim_data[C.DATA_KEY_INPUT_TENSORS_SCALED]
        target_tensors_scaled = sim_data[C.DATA_KEY_TARGET_TENSORS_SCALED]

        if override_config:
            merged_config = {**common_parameters, **base_configurations[0]}
            sim_data[C.DATA_KEY_MERGED_CONFIG] = merged_config
            base_configuration = base_configurations[0]
            sim_data[C.DATA_KEY_BASE_CONFIGURATION] = base_configuration
        else:
            sim_data[C.DATA_KEY_BASE_CONFIGURATION] = base_configuration[0]

        # Override 'figs_save_dir' path with the new path if loading from disk
        sim_data[C.DATA_KEY_MERGED_CONFIG]["figs_save_dir"] = SAVE_DIR_FIGS
        sim_data[C.DATA_KEY_BASE_CONFIGURATION]["figs_save_dir"] = SAVE_DIR_FIGS

        # Check if data was collected
        if input_tensors_scaled is not None and target_tensors_scaled is not None:
            print(f"All Input Tensors Shape: {input_tensors_scaled.shape}")
            print(f"All Target Tensors Shape: {target_tensors_scaled.shape}")
        else:
            print("No simulations within the specified delta range.")
            exit(1)  # Exit if no data was collected

    return sim_data


def get_data_sub_cfg(sim_data):
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
    return data_sub_cfg


def add_xy_coupling(input_tensors):
    """Add x*y coupling feature to input data.

    Args:
        input_tensors: Tensor of shape (n_samples, n_turns, n_BPMs, n_planes) where n_planes=2 for x,y

    Returns:
        Tensor of shape (n_samples, n_turns, n_BPMs, 3) with x, y, x*y
    """
    x_plane = input_tensors[..., 0:1]
    y_plane = input_tensors[..., 1:2]
    xy_coupling = x_plane * y_plane
    return torch.cat([x_plane, y_plane, xy_coupling], dim=-1)


def get_data_splits(sim_data, test_size=0.10, model_arch=None, couple_xy=False):
    """Get data splits

    Args:
        sim_data: Simulation data dictionary
        test_size: Fraction of data to use for validation
        model_arch: Model architecture identifier
        couple_xy: If True, compute x*y coupling feature (for LSTM and CNN1D)
    """
    if model_arch is None:
        raise ValueError("Model architecture must be specified.")

    if test_size < 0 or test_size > 1:
        raise ValueError("Test size must be between 0 and 1.")

    input_tensors_scaled = sim_data[C.DATA_KEY_INPUT_TENSORS_SCALED]
    target_tensors_scaled = sim_data[C.DATA_KEY_TARGET_TENSORS_SCALED]

    raw_input_tensors = input_tensors_scaled
    raw_target_tensors = target_tensors_scaled

    n_samples, n_turns, n_BPMs, n_planes = raw_input_tensors.shape

    if model_arch == C.NET_ARCH_CNN1D:
        input_with_coupling = add_xy_coupling(raw_input_tensors)
        print(f"CNN1D Input Data Shape with xy-coupling: {input_with_coupling.shape}")
        input_data_np = input_with_coupling.numpy()
    elif couple_xy:
        input_with_coupling = add_xy_coupling(raw_input_tensors)
        print(f"LSTM Input Data Shape with xy-coupling: {input_with_coupling.shape}")
        input_data_np = input_with_coupling.numpy()
    else:
        input_size = n_BPMs * n_planes
        input_data_np = raw_input_tensors.reshape(
            n_samples, n_turns, input_size
        ).numpy()

    target_data = raw_target_tensors

    target_data_np = target_data.numpy()

    train_inputs, val_inputs, train_targets, val_targets = train_test_split(
        input_data_np, target_data_np, test_size=test_size, shuffle=True
    )

    data_shapes = {
        "inputs_shape": train_inputs.shape,
        "targets_shape": train_targets.shape,
        "raw_input_tensors_shape": raw_input_tensors.shape,
    }

    return train_inputs, val_inputs, train_targets, val_targets, data_shapes


def prepare_data_for_training(
    sim_data, test_size=0.10, batch_size=16, model_arch=None, couple_xy=False
):
    """Prepare data for training"""
    train_inputs, val_inputs, train_targets, val_targets, data_shapes = get_data_splits(
        sim_data, test_size, model_arch, couple_xy=couple_xy
    )

    # Convert back to tensors
    train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
    val_inputs = torch.tensor(val_inputs, dtype=torch.float32)
    train_targets = torch.tensor(train_targets, dtype=torch.float32)
    val_targets = torch.tensor(val_targets, dtype=torch.float32)

    # Create datasets and data loaders
    train_dataset = TensorDataset(train_inputs, train_targets)
    val_dataset = TensorDataset(val_inputs, val_targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, data_shapes
