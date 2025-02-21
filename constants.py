# constants.py

class Constants:
    DATA_KEY_ALL_ERROR_VALUES_DIPOLE_TILT = 'all_error_values_dipole_tilt'
    DATA_KEY_ALL_ERROR_VALUES_QUAD_MISALIGN = 'all_error_values_quad_misalign'
    DATA_KEY_ALL_ERROR_VALUES_QUAD_TILT = 'all_error_values_quad_tilt'
    DATA_KEY_DATA_AUTOMATION = 'data_automation'
    DATA_KEY_DATASET_SCALERS = 'dataset_scalers'
    DATA_KEY_INPUT_TENSORS = 'input_tensors'
    DATA_KEY_INPUT_TENSORS_SCALED = 'input_tensors_scaled'
    DATA_KEY_MERGED_CONFIG = 'merged_config'
    DATA_KEY_BASE_CONFIGURATION = 'base_configuration'
    DATA_KEY_TARGET_TENSORS = 'target_tensors'
    DATA_KEY_TARGET_TENSORS_SCALED = 'target_tensors_scaled'
    DATA_KEY_TARGET_SCALER = 'target_scaler'
    
    KEY_ACCUMULATED_DATASETS = 'accumulated_datasets'
    
    NET_ARCH_SIMPLE_FULLY_CONNECTED = 'SimpleFullyConnectedNetwork'
    NET_ARCH_SIMPLE_CNN = 'SimpleCNN'
    NET_ARCH_LSTM = 'QuadErrorCorrectionLSTM'
