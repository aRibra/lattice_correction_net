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


def gen_data(n_simulations = 0):
    """Data Generation using DataAutomation
    Data is saved automatically. The method DataAutomation.save_data() is called after each simualtion
    to save accumulated data while runing. This prevents losing all the data when running large number of
    simualtions.

    Args:
        n_simulations (int, optional): Defaults to 0.
    """
    
    # Create an instance of DataAutomation with delta_range
    data_automation = DataAutomation(base_configurations, common_parameters, n_simulations)
    dataset_scalers = data_automation.get_data_scalers()
    
    data_tag = data_automation.tag
    print("Using tag:", data_tag)
    
    # Run the data automation process and get the data tensors
    data_automation.run(
        include_no_error_data=False, 
        skip_data_on_delta_ranges=False)
    
    data_dir = os.path.join(f'data/{data_tag}')
    return load_data_from_dir(data_dir=data_dir)


def load_data_from_dir(data_dir=None, override_config=True):
    """Load data from a directory
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError("Data directory does not exist.")
    
    if data_dir:
        postfix = 'final'

        sim_data = {}

        sim_data[C.DATA_KEY_ALL_ERROR_VALUES_DIPOLE_TILT] =   torch.load(f'{data_dir}/{C.DATA_KEY_ALL_ERROR_VALUES_DIPOLE_TILT}-{postfix}.pt')
        sim_data[C.DATA_KEY_ALL_ERROR_VALUES_QUAD_MISALIGN] = torch.load(f'{data_dir}/{C.DATA_KEY_ALL_ERROR_VALUES_QUAD_MISALIGN}-{postfix}.pt')
        sim_data[C.DATA_KEY_ALL_ERROR_VALUES_QUAD_TILT] =     torch.load(f'{data_dir}/{C.DATA_KEY_ALL_ERROR_VALUES_QUAD_TILT}-{postfix}.pt')
        sim_data[C.DATA_KEY_DATA_AUTOMATION] =                torch.load(f'{data_dir}/{C.DATA_KEY_DATA_AUTOMATION}-{postfix}.pt')
        sim_data[C.DATA_KEY_DATASET_SCALERS] =                torch.load(f'{data_dir}/{C.DATA_KEY_DATASET_SCALERS}-{postfix}.pt')
        sim_data[C.DATA_KEY_INPUT_TENSORS] =                  torch.load(f'{data_dir}/{C.DATA_KEY_INPUT_TENSORS}-{postfix}.pt')
        sim_data[C.DATA_KEY_INPUT_TENSORS_SCALED] =           torch.load(f'{data_dir}/{C.DATA_KEY_INPUT_TENSORS_SCALED}-{postfix}.pt')
        sim_data[C.DATA_KEY_MERGED_CONFIG] =                  torch.load(f'{data_dir}/{C.DATA_KEY_MERGED_CONFIG}-{postfix}.pt')
        sim_data[C.DATA_KEY_TARGET_TENSORS] =                 torch.load(f'{data_dir}/{C.DATA_KEY_TARGET_TENSORS}-{postfix}.pt')
        sim_data[C.DATA_KEY_TARGET_TENSORS_SCALED] =          torch.load(f'{data_dir}/{C.DATA_KEY_TARGET_TENSORS_SCALED}-{postfix}.pt')

        data_automation = sim_data[C.DATA_KEY_DATA_AUTOMATION]

        base_configuration = [data_automation.overridden_base_config]
        input_tensors_scaled = sim_data[C.DATA_KEY_INPUT_TENSORS_SCALED]
        target_tensors_scaled = sim_data[C.DATA_KEY_TARGET_TENSORS_SCALED]
        
        if override_config:
            merged_config = {**common_parameters, **base_configurations[0]}
            sim_data[C.DATA_KEY_MERGED_CONFIG] = merged_config
            base_configuration = base_configurations[0]
            
        sim_data[C.DATA_KEY_BASE_CONFIGURATION] = base_configuration

        # Override 'figs_save_dir' path with the new path if loading from disk
        sim_data[C.DATA_KEY_MERGED_CONFIG]['figs_save_dir'] = SAVE_DIR_FIGS
        sim_data[C.DATA_KEY_BASE_CONFIGURATION]['figs_save_dir'] = SAVE_DIR_FIGS

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
        'merged_config': sim_data[C.DATA_KEY_MERGED_CONFIG],
        'input_scaler_config': serialize_minmax_scaler(sim_data[C.DATA_KEY_DATASET_SCALERS]['input_scaler']),
        'target_scaler_config': serialize_minmax_scaler(sim_data[C.DATA_KEY_DATASET_SCALERS]['target_scaler']),
        'overridden_base_config': sim_data[C.DATA_KEY_DATA_AUTOMATION].overridden_base_config.copy()
    }
    return data_sub_cfg


def get_data_splits(sim_data, test_size=0.10, model_arch=None):
    """Get data splits
    """
    if model_arch is None:
        raise ValueError("Model architecture must be specified.")
    
    if test_size < 0 or test_size > 1:
        raise ValueError("Test size must be between 0 and 1.")
    
    input_tensors_scaled = sim_data[C.DATA_KEY_INPUT_TENSORS_SCALED]
    target_tensors_scaled = sim_data[C.DATA_KEY_TARGET_TENSORS_SCALED]
    
    raw_input_tensors = input_tensors_scaled # input_tensors_scaled | input_tensors
    raw_target_tensors = target_tensors_scaled # target_tensors_scaled | target_tensors

    if model_arch == C.NET_ARCH_SIMPLE_CNN:
        input_data_cnn = raw_input_tensors.permute(0, 3, 1, 2)  # Shape: (n_samples, 2, 101, N_FODO)
        print(f"CNN Input Data Shape: {input_data_cnn.shape}")  # Verify the shape

    # Convert input data to appropriate shape
    n_samples, n_turns, n_BPMs, n_planes = raw_input_tensors.shape
    input_size = n_BPMs * n_planes
    input_data = raw_input_tensors.reshape(n_samples, n_turns, input_size)

    # Convert target data to appropriate shape
    target_data = raw_target_tensors  # Shape: (n_samples, n_errors), where n_errors = 1

    # Split data into training and validation sets
    input_data_np = input_data.numpy()
    target_data_np = target_data.numpy()

    train_inputs, val_inputs, train_targets, val_targets = train_test_split(
        input_data_np, target_data_np, test_size=test_size, shuffle=True) #random_state=42

    data_shapes = {
        'inputs_shape': train_inputs.shape,
        'targets_shape': train_targets.shape, 
        'raw_input_tensors_shape': raw_input_tensors.shape
    }

    return train_inputs, val_inputs, train_targets, val_targets, data_shapes


def prepare_data_for_training(sim_data, test_size=0.10, batch_size=16, model_arch=None):
    """Prepare data for training
    """
    train_inputs, val_inputs, train_targets, val_targets, data_shapes = get_data_splits(sim_data, test_size, model_arch)

    # Convert back to tensors
    train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
    val_inputs = torch.tensor(val_inputs, dtype=torch.float32)
    train_targets = torch.tensor(train_targets, dtype=torch.float32)
    val_targets = torch.tensor(val_targets, dtype=torch.float32)

    # Create datasets and data loaders
    train_dataset = TensorDataset(train_inputs, train_targets)
    val_dataset = TensorDataset(val_inputs, val_targets)
    
    n_samples, n_turns, input_size = train_inputs.shape
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, data_shapes

