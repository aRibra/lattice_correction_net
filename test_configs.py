
## This file contains the configuration parameters for benchmark and consistency test simulations across backends.

config = {
    'design_radius': 20.0,
    'L_quad': 0.4,
    'L_straight': 3.9269908169872423,
    'q': 1.602e-19,
    'p': 5.344286e-19,
    "n_turns":1000,
    'total_dipole_bending_angle': 4.71238898038469,
    "num_particles":100000,
    'n_FODO': 8,
    'L_dipole': 5.890486225480862,
    'n_Dipoles': 16,
    'G': 1.0,
    'f': 3.336008739076155,
    "use_thin_lens":True,
    "mag_field_range":[
        0.1,
        2.0
    ],
    "dipole_length_range":[
        0.2,
        14.0
    ],
    "horizontal_tune_range":[
        0.1,
        0.8
    ],
    "vertical_tune_range":[
        0.1,
        0.8
    ],
    "verbose":True,
    "correct_injection_offset":False,
    "max_iter_per_infer":1,


}

other_config = {
    "p":5.344286e-19,
    "G":1.0,
    "q":1.602e-19,
    "window_size":10,
    "x0_mean_std":(0.0,
    0.05),
    "xp0_mean_std":(0.0,
    0.0),
    "y0_mean_std":(0.0,
    0.05),
    "yp0_mean_std":(0.0,
    0.0),
    "particles_sampling_method":"from_twiss_params",
    "sampling_circle_radius":0.01,
    "total_dipole_bending_angle_range":(4.71238898038469,
    6.283185307179586),
    "target_data":"quad_misalign_deltas",
    "reject_multiple_error_types":False,
    "start_rev":499,
    "end_rev":-1,
    "apply_avg":False,
    "fodo_cell_indices":[
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7
    ],
    "planes":[
        "x",
        "y"
    ],
    "random_criterion":"normal",
    "delta_range":(0,
    5e-05),
    "quad_tilt_angle_range":(0,
    0.01),
    "dipole_tilt_angle_range":(0,
    0.05),
    "com_delta_range":(-5e-05,
    5e-05),
    "config_name":"Configuration 1",
    "design_radius":20.0,
    "n_FODO":8,
    "f":3.336008739076155,
    "L_quad":0.4,
    "quad_errors":"None",
    "quad_tilt_errors":"None",
    "dipole_tilt_errors":"None",
    "L_straight":3.9269908169872423,
    "L_dipole":5.890486225480862,
    "n_Dipoles":16,
    "L_drift":0.7817477042468106,
    "Qx":0.4677523855413337,
    "Qy":0.17971989927789359,
    "B":0.16680043695380775,
    "B_rho":3.336008739076155,
    "total_dipole_bending_angle":4.71238898038469
}
