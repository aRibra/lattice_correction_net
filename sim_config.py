# config file for main.py script

import os
import numpy as np
from datetime import datetime

EXPS_FOLDER = 'exps'
TAG = 'test'

SHOW_PLOTS = False

# Create a directory for the experiment session with the current date and time
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

RUN_TAG = f'{EXPS_FOLDER}/{TAG}/run_{timestamp}'

SAVE_DIR_FIGS = f'{RUN_TAG}/figs'
SAVE_DIR_TRAINING = f'{RUN_TAG}/training'
SAVE_DIR_BENCHMARKS = f'{RUN_TAG}/benchmarks'

for dir in [SAVE_DIR_FIGS, SAVE_DIR_TRAINING, SAVE_DIR_BENCHMARKS]:
    print("--->Creating save directory:", dir)
    os.makedirs(dir, exist_ok=True)


# Define base simulation configurations with embedded quadrupole errors
base_configurations = [
    {
        'config_name': 'Configuration 1',
        'design_radius': 20.0,    # meters
        'n_FODO': 8,
        'f': 15,                 # meters
        'L_quad': 0.4,             # meters
        # 'L_straight': 1,         # meters
        # 'total_dipole_bending_angle': (2 * np.pi),#(3 / 3) * np.pi,
        
        # 'quad_errors': None,

        'quad_errors': [
            # {
            #     'FODO_index': 0,
            #     'quad_type': 'defocusing',
            #     'delta': 1e-5,
            #     'plane': 'vertical'  # 'horizontal' or 'vertical'
            # },
            {
                'FODO_index': 1,
                'quad_type': 'defocusing',
                'delta': 1e-6,
                'plane': 'vertical'  # 'horizontal' or 'vertical'
            },
            {
                'FODO_index': 2,
                'quad_type': 'defocusing',
                'delta': 1e-7,
                'plane': 'vertical'  # 'horizontal' or 'vertical'
            },
            {
                'FODO_index': 3,
                'quad_type': 'defocusing',
                'delta': 1e-7,
                'plane': 'vertical'  # 'horizontal' or 'vertical'
            },
            {
                'FODO_index': 4,
                'quad_type': 'defocusing',
                'delta': 1e-7,
                'plane': 'vertical'  # 'horizontal' or 'vertical'
            },
            {
                'FODO_index': 5,
                'quad_type': 'defocusing',
                'delta': 1e-7,
                'plane': 'vertical'  # 'horizontal' or 'vertical'
            },
            # {
            #     'FODO_index': 6,
            #     'quad_type': 'defocusing',
            #     'delta': 1e-7,
            #     'plane': 'vertical'  # 'horizontal' or 'vertical'
            # },
            {
                'FODO_index': 7,
                'quad_type': 'defocusing',
                'delta': 1e-7,
                'plane': 'vertical'  # 'horizontal' or 'vertical'
            },
        ],

        # "quad_tilt_errors": None,
        
        'quad_tilt_errors': [
            {
                'FODO_index': 1,
                'quad_type': 'defocusing',
                'tilt_angle': 0.0005
            },
            {
                'FODO_index': 3,
                'quad_type': 'defocusing',
                'tilt_angle': 0.0005
            },
            {
                'FODO_index': 6,
                'quad_type': 'defocusing',
                'tilt_angle': 0.0005
            },   
        ],

        'dipole_tilt_errors': None,
        # 'dipole_tilt_errors': [
        #     {
        #         'FODO_index': 0,
        #         'dipole_index': 0,
        #         'tilt_angle': 0.005
        #     },
        #     {
        #         'FODO_index': 1,
        #         'dipole_index': 1,
        #         'tilt_angle': 0.005
        #     },
        #     {
        #         'FODO_index': 7,
        #         'dipole_index': 1,
        #         'tilt_angle': 0.005
        #     }
        # ],

    },
]

common_parameters = {
    'p': 5.344286e-19,                # Momentum in kg m/s (p_GeV_c=0.7)
    'G': 1.0,                         # Tesla/meter
    'q': 1.602e-19,                   # Proton charge in Coulombs
    'n_turns': 5000,                   # Number of revolutions to simulate
    'num_particles': 5,             # Number of particles to simulate
    'window_size': 10,                # Average window size for moving averages
    'use_thin_lens': True,
    # Uniform - Initial conditions ranges as tuples
    'x0_mean_std': (0.0, 0.05),      # meters
    'xp0_mean_std': (0.0, 0.00),  # radians
    'y0_mean_std': (0.0, 0.05),      # meters
    'yp0_mean_std': (0.0, 0.00),  # radians
    'particles_sampling_method': 'from_twiss_params', # from_twiss_params | circle_with_radius | normal
    'sampling_circle_radius': 0.01, #meters,
    # Acceptable ranges config params
    'mag_field_range': [0.1, 2.0],           # Tesla
    'dipole_length_range': [0.2, 14.0],       # meters
    'horizontal_tune_range': [0.1, 0.8],     # Tune
    'vertical_tune_range': [0.1, 0.8],        # Tune
    'total_dipole_bending_angle_range': (1.5 * np.pi, 2 * np.pi),
    # Use cuda GPU kernels to accelerate simulation
    'use_gpu': True,
    # Log
    'verbose': True,
    # BPM readings log criterion
    # record_full_revolution. By default records for cell_idx=0
    # 'record_full_revolution': False,
    # enable storing BPM readings for all BPMs after each full revolution relative to each BPM
    # requires to set record_full_revolution to `True`
    # 'record_full_revolution_per_bpm': False,
    
    'figs_save_dir': f'{SAVE_DIR_FIGS}',
    
    # Parameters for generate_data
    'target_data': 'quad_misalign_deltas',  # ['quad_misalign_deltas', 'quad_tilt_angles', 'dipole_tilt_angles']
    'reject_multiple_error_types': False,
    'start_rev': 4500,
    'end_rev': 5000, # should be same as <= n_turns
    'apply_avg': True,
    'fodo_cell_indices': [0, 1, 2, 3, 4, 5, 6, 7],  # Indices of BPMs to consider
    'planes': ['x', 'y'],
    'random_criterion': 'normal', # uniform | normal
    # if 'random_criterion' is 'uniform'
        # then >> _range is (low, high)
    # if 'random_criterion' is 'normal'
        # then >> _range is (mean, std)
    'delta_range': (0, 5e-5),
    'quad_tilt_angle_range': (0, 0.010), # 10 mrad (1e-3 = 1 mrad)
    'dipole_tilt_angle_range': (0, 0.05), # 50 mrad
    'com_delta_range': (-5e-05, 5e-05),
}
