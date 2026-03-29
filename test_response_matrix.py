import numpy as np
import torch
from synchrotron_simulator_gpu_Dataset_4D import SynchrotronSimulator

def test_response_matrix():
    print("==================================================")
    print("   ANALYTICAL R-MATRIX VS. EXACT CLOSED ORBIT     ")
    print("==================================================")

    # 1. Define ONE specific error: 1 mm offset on FODO 1 Defocusing Quad
    delta_y_offset = 0.001  # 1 mm in meters
    test_quad_cfg = [
        {
            "FODO_index": 1,
            "quad_type": "defocusing",
            "delta": delta_y_offset,
            "plane": "vertical",  # 'horizontal' or 'vertical'
        }
    ]

    # 2. Initialize a CLEAN Simulator
    # Ensure all random noise (K-drifts, tilt errors) are disabled!

    merged_config = torch.load("data/Sim2000_6000turns_300parts_FODOErr-123457-136-_avgFalse_tgtquad_misalign_deltas_1/merged_config-final.pt")  # Load the merged configuration dictionary

    merged_config['k_errors'] = None
    merged_config['quad_tilt_errors'] = None
    merged_config['quad_errors'] = None
    merged_config['dipole_tilt_errors'] = None

    sim = SynchrotronSimulator(
        design_radius=merged_config["design_radius"],
        G=merged_config["G"],
        f=merged_config["f"],
        use_thin_lens=merged_config["use_thin_lens"],
        L_quad=merged_config["L_quad"],
        L_straight=merged_config["L_straight"],
        p=merged_config["p"],
        q=merged_config["q"],
        total_dipole_bending_angle=merged_config["total_dipole_bending_angle"],
        dipole_length_range=merged_config["dipole_length_range"],
        num_particles=merged_config["num_particles"],
        n_FODO=merged_config["n_FODO"],
        L_dipole=merged_config["L_dipole"],
        n_Dipoles=merged_config["n_Dipoles"],
        mag_field_range=merged_config["mag_field_range"],
        horizontal_tune_range=merged_config["horizontal_tune_range"],
        vertical_tune_range=merged_config["vertical_tune_range"],
        n_turns=1,  # We only need 1 turn to build the lattice and grab the matrices
        verbose=True
    )

    # Inject the known error and build the lattice matrices
    
    # Introduce quadrupole misalignment errors
    for qerr in test_quad_cfg:
        print('Injecting Quad Error:', qerr)
        sim.set_quad_error(
            FODO_index=qerr["FODO_index"],
            quad_type=qerr["quad_type"],
            delta=qerr["delta"],
            plane=qerr["plane"],
        )
    sim.build_lattice()
    sim.compute_tunes()

    # ---------------------------------------------------------
    # PART A: The Analytical Prediction (R-Matrix)
    # ---------------------------------------------------------
    # R_y shape will be (8, 1) because we are only passing 1 target quad
    R_y_tensor = sim.compute_response_matrix_y(quad_errors_cfg=test_quad_cfg, device='cpu')
    R_y = R_y_tensor.numpy()
    
    dy_quad = np.array([[delta_y_offset]])  # +1 mm offset
    analytical_bpm_y = R_y @ dy_quad # [8x1] @ [1x1] = [8x1]

    # ---------------------------------------------------------
    # PART B: The Exact Simulated Orbit (M*X + D)
    # ---------------------------------------------------------
    # Get the 4D closed orbit at the start of the ring (s=0)
    x_co, y_co = sim.compute_closed_orbit()
    print("====> Closed Orbit at s=0 (x, x', y, y'):", x_co[0], x_co[1], y_co[0], y_co[1])
    X_current = np.array([x_co[0], x_co[1], y_co[0], y_co[1]])

    simulated_bpm_y = []
    global_idx = 0

    # Propagate the closed orbit element-by-element to every BPM
    for cell_idx, n_elems in enumerate(sim.len_per_cell_list):
        # Record the Y position at the entrance of the cell (The BPM)
        simulated_bpm_y.append(X_current[2])
        
        # Transport the beam through this cell's elements
        for _ in range(n_elems):
            M = sim.M_lattice_4x4[global_idx]
            D = sim.D_lattice_4x1[global_idx]
            X_current = M @ X_current + D
            global_idx += 1

    # ---------------------------------------------------------
    # PART C: The Comparison
    # ---------------------------------------------------------
    print(f"{'BPM':<5} | {'Analytical R_y (m)':<20} | {'Simulated Orbit (m)':<20} | {'Error (%)':<10}")
    print("-" * 65)
    
    for i in range(8):
        ana = analytical_bpm_y[i, 0]
        sim_val = simulated_bpm_y[i]
        
        # Calculate percentage difference
        diff = abs(ana - sim_val) / (abs(ana) + 1e-12) * 100
        
        print(f"{i:<5} | {ana:>18.8f} | {sim_val:>18.8f} | {diff:>8.4f}%")

if __name__ == "__main__":
    test_response_matrix()