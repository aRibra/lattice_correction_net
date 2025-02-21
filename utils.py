# utils.py

import json
from collections import defaultdict

from sklearn.preprocessing import MinMaxScaler


def convert_defaultdict_to_dict(d):
    """
    Recursively converts a defaultdict to a regular dict.

    Parameters:
    - d: The defaultdict or dict to convert.

    Returns:
    - A standard dict with all nested defaultdicts converted.
    """
    if isinstance(d, defaultdict):
        d = {k: convert_defaultdict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, dict):
        d = {k: convert_defaultdict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        d = [convert_defaultdict_to_dict(item) for item in d]
    return d


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def serialize_minmax_scaler(scaler):
    """
    Serialize a MinMaxScaler to a JSON file.

    Parameters:
    - scaler: MinMaxScaler object to serialize.
    """
    scaler_params = {
        'min_': scaler.min_.tolist(),
        'scale_': scaler.scale_.tolist(),
        'data_min_': scaler.data_min_.tolist(),
        'data_max_': scaler.data_max_.tolist(),
        'data_range_': scaler.data_range_.tolist(),
        'feature_range': scaler.feature_range
    }
    
    return scaler_params


def deserialize_minmax_scaler(scaler_params):
    """
    Deserialize a MinMaxScaler from a JSON file.

    Parameters:
    - filepath: Path to the JSON file from which the scaler will be loaded.

    Returns:
    - MinMaxScaler object reconstructed from the JSON file.
    """
    scaler = MinMaxScaler(feature_range=tuple(scaler_params['feature_range']))
    scaler.min_ = scaler_params['min_']
    scaler.scale_ = scaler_params['scale_']
    scaler.data_min_ = scaler_params['data_min_']
    scaler.data_max_ = scaler_params['data_max_']
    scaler.data_range_ = scaler_params['data_range_']
    
    return scaler
