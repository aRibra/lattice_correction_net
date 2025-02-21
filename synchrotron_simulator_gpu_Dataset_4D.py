# synchrotron_simulator_gpu_Dataset_4D.py

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import traceback
from numba import njit, prange
from numba import cuda, float64
# import torch
import matplotlib.cm as cm



# Define color palette for plotting
prop_cycle = plt.rcParams['axes.prop_cycle']
base_colors_cycle = prop_cycle.by_key()['color']

def split_merged_config(merged_config):
    # base_configurations keys
    base_keys = {
        'config_name',
        'design_radius',
        'n_FODO',
        'f',
        'L_quad',
        'L_straight',
        'quad_errors',
        'quad_tilt_errors',
        'dipole_tilt_errors'
    }

    # Extract base configuration parameters
    base_config = {k: v for k, v in merged_config.items() if k in base_keys}

    # wrap the base_config in a list
    base_configurations = [base_config]

    # extract common parameters and exclude base_keys
    common_parameters = {k: v for k, v in merged_config.items() if k not in base_keys}

    return base_configurations, common_parameters


def get_colors(n_particles):
    repeat_times = int(np.ceil(n_particles / len(base_colors_cycle)))
    colors = base_colors_cycle * repeat_times
    return colors[:n_particles]


def drift_4x4(L):
    """
    Return the 4x4 transfer matrix for a drift of length L.
    (x -> x + L*x', y -> y + L*y')
    """
    M = np.identity(4, dtype=np.float64)
    M[0,1] = L
    M[2,3] = L
    return M

def quad_transfer_matrix_4x4(k, L, thin_lens=False):
    """
    Return the 4x4 transfer matrix for a quadrupole of focal strength k (1/m^2) 
    and length L, either thick-lens or thin-lens.
    - If k > 0 => focusing in x-plane, defocusing in y-plane
    - If k < 0 => defocusing in x-plane, focusing in y-plane
    """
    if thin_lens:
        # Thin-lens approximation: M = Identity + Kick
        M = np.identity(4, dtype=np.float64)
        # x-plane focal kick: x' -> x' - k*L*x
        # y-plane defocusing: y' -> y' + k*L*y
        # This is a linear approximation ignoring cross-plane coupling beyond sign swap:
        M[1,0] = -k*L   # focusing in x
        M[3,2] =  k*L   # defocusing in y
        return M

    # --- Thick-lens quadrupole: block-by-block approach ---
    M = np.identity(4, dtype=np.float64)

    # x-plane sub-block (rows 0-1, cols 0-1) for +k
    kx = k  # focusing if k>0
    if abs(kx) > 1e-16:
        sqrt_kx = np.sqrt(abs(kx))
        if kx > 0:
            # focusing in x
            M[0,0] = np.cos(sqrt_kx*L)
            M[0,1] = np.sin(sqrt_kx*L)/sqrt_kx
            M[1,0] = -sqrt_kx*np.sin(sqrt_kx*L)
            M[1,1] = np.cos(sqrt_kx*L)
        else:
            # defocusing in x => cosh, sinh
            M[0,0] = np.cosh(sqrt_kx*L)
            M[0,1] = np.sinh(sqrt_kx*L)/sqrt_kx
            M[1,0] = sqrt_kx*np.sinh(sqrt_kx*L)
            M[1,1] = np.cosh(sqrt_kx*L)
    else:
        # k ~ 0 => drift
        M[0,1] = L

    # y-plane sub-block (rows 2-3, cols 2-3) for -k
    ky = -k
    if abs(ky) > 1e-16:
        sqrt_ky = np.sqrt(abs(ky))
        if ky > 0:
            # focusing in y
            M[2,2] = np.cos(sqrt_ky*L)
            M[2,3] = np.sin(sqrt_ky*L)/sqrt_ky
            M[3,2] = -sqrt_ky*np.sin(sqrt_ky*L)
            M[3,3] = np.cos(sqrt_ky*L)
        else:
            # defocusing in y => cosh, sinh
            M[2,2] = np.cosh(sqrt_ky*L)
            M[2,3] = np.sinh(sqrt_ky*L)/sqrt_ky
            M[3,2] = sqrt_ky*np.sinh(sqrt_ky*L)
            M[3,3] = np.cosh(sqrt_ky*L)
    else:
        # ky ~ 0 => drift
        M[2,3] = L

    return M

def pure_sector_dipole_4x4(L, rho):
    """
    4x4 transfer matrix for a pure sector dipole (no quadrupole component).
    
    Parameters:
        L (float):   Length of the dipole [m].
        rho (float): Bending radius [m].
        
    Returns:
        M_4x4 (ndarray): 4x4 transfer matrix.
        theta (float):    Bending angle [rad] (theta = L / rho).
        D4 (ndarray):     Zero 4-vector (no inhomogeneous terms).
    """
    theta = L / rho  # Bending angle
    
    # Horizontal plane (bending dynamics)
    Mx = np.array([
        [np.cos(theta),       rho * np.sin(theta)],
        [-np.sin(theta)/rho,  np.cos(theta)     ]
    ], dtype=np.float64)
    
    # Vertical plane (drift)
    My = np.array([
        [1.0, L],
        [0.0, 1.0]
    ], dtype=np.float64)
    
    # Block-diagonal 4x4 matrix
    M_4x4 = np.zeros((4, 4), dtype=np.float64)
    M_4x4[0:2, 0:2] = Mx  # Horizontal block
    M_4x4[2:4, 2:4] = My  # Vertical block
    
    # No dipole kick (homogeneous solution)
    D4 = np.zeros(4, dtype=np.float64)
    
    return M_4x4

def tilt_4x4(phi):
    """
    Return a 4x4 rotation matrix for a tilt by angle phi about the beam axis, coupling (x,x') and (y,y').
    """
    c = np.cos(phi)
    s = np.sin(phi)
    return np.array([
        [ c,  0,  s,  0],
        [ 0,  c,  0,  s],
        [-s, 0,  c,  0],
        [ 0,-s,  0,  c]
    ], dtype=np.float64)

def make_tilt_element_4x4(tilt_angle, description="Tilt Error"):
    """
    Create a zero-length element representing a magnet tilt by `tilt_angle`.
    M_4x4 = tilt_4x4(tilt_angle).
    """
    return {
        'element_type': 'TiltError',
        'description': description,
        'M_4x4': tilt_4x4(tilt_angle),
        's_elem': 0.0,
    }


class QuadrupoleMisAlignError:
    def __init__(self, FODO_index, quad_type, delta, plane):
        """
        Initialize a QuadrupoleMisAlignError instance.

        Parameters:
            FODO_index (int): Index of the FODO cell where the error is introduced (0-based).
            quad_type (str): 'focusing' or 'defocusing'.
            delta (float): Displacement error (meters).
            plane (str): 'horizontal' or 'vertical' indicating which plane to apply the error.
        """
        if delta > 1:
            raise ValueError(f"Quadrupole displacement delta={delta} m exceeds the stability limit of 1e-4 m.")
        if quad_type.lower() not in ['focusing', 'defocusing']:
            raise ValueError("quad_type must be either 'focusing' or 'defocusing'.")
        if plane.lower() not in ['horizontal', 'vertical']:
            raise ValueError("plane must be either 'horizontal' or 'vertical'.")
        self.FODO_index = FODO_index
        self.quad_type = quad_type.lower()
        self.delta = delta
        self.plane = plane.lower()

class DipoleTiltError:
    def __init__(self, FODO_index, dipole_index, tilt_angle):
        """
        FODO_index: which FODO cell to apply tilt
        dipole_index: e.g. 0 for first dipole, 1 for second dipole in that cell
        tilt_angle: rad
        """
        self.FODO_index = FODO_index
        self.dipole_index = dipole_index
        self.tilt_angle = tilt_angle

class QuadrupoleTiltError:
    def __init__(self, FODO_index, quad_type, tilt_angle):
        """
        FODO_index: which FODO cell
        quad_type: 'focusing' or 'defocusing'
        tilt_angle: rad
        """
        self.FODO_index = FODO_index
        self.quad_type = quad_type.lower()
        self.tilt_angle = tilt_angle


class FODOCell4D:
    """
    Basic FODO cell: drift -> dipole -> drift -> quad -> ...
    returning 4x4 matrices for each element
    """
    def __init__(self, L_quad, L_drift, L_dipole, theta_dipole,
                 k_f, k_d, rho, k_nominal, use_thin_lens=False):
        self.L_quad   = L_quad
        self.L_drift  = L_drift
        self.L_dipole = L_dipole
        self.theta_dipole = theta_dipole
        self.k_f = k_f
        self.k_d = k_d
        self.rho = rho
        self.k_nominal = k_nominal
        self.use_thin_lens = use_thin_lens

    def get_elements_4x4(self):
        """
        Return a list of dicts, each with:
           {
             'element_type': str,
             'description': str,
             'M_4x4': np.array((4,4)),
             's_elem': float,
           }
        for one FODO cell. E.g.:
          - Drift
          - Dipole #0
          - Drift
          - Defocusing Quad
          - Drift
          - Dipole #1
          - Drift
          - Focusing Quad
        """
        elements = []

        # 1) Drift
        Md = drift_4x4(self.L_drift)
        elements.append({
            'element_type': 'Drift',
            'description': 'Drift Before Dipole #0',
            'M_4x4': Md,
            'D_4x1': np.zeros(4, dtype=np.float64),
            's_elem': self.L_drift
        })

        # 2) Dipole #0
        M_d0 = pure_sector_dipole_4x4(self.L_dipole, self.rho)

        elements.append({
            'element_type': 'Dipole',
            'description': 'Dipole #0',
            'M_4x4': M_d0,
            'D_4x1': np.zeros(4, dtype=np.float64),
            's_elem': self.L_dipole
        })

        # 3) Drift
        Md = drift_4x4(self.L_drift)
        elements.append({
            'element_type': 'Drift',
            'description': 'Drift After Dipole #0',
            'M_4x4': Md,
            'D_4x1': np.zeros(4, dtype=np.float64),
            's_elem': self.L_drift
        })

        # 4) Quad (Defocusing)
        M_qd = quad_transfer_matrix_4x4(self.k_d, self.L_quad, self.use_thin_lens)
        elements.append({
            'element_type': 'Quad',
            'description': 'Defocusing Quad',
            'M_4x4': M_qd,
            'D_4x1': np.zeros(4, dtype=np.float64),
            's_elem': self.L_quad
        })

        # 5) Drift
        Md = drift_4x4(self.L_drift)
        elements.append({
            'element_type': 'Drift',
            'description': 'Drift Before Dipole #1',
            'M_4x4': Md,
            'D_4x1': np.zeros(4, dtype=np.float64),
            's_elem': self.L_drift
        })

        # 6) Dipole #1
        M_d1 = pure_sector_dipole_4x4(self.L_dipole, self.rho)

        elements.append({
            'element_type': 'Dipole',
            'description': 'Dipole #1',
            'M_4x4': M_d1,
            'D_4x1': np.zeros(4, dtype=np.float64),
            's_elem': self.L_dipole
        })

        # 7) Drift
        Md = drift_4x4(self.L_drift)
        elements.append({
            'element_type': 'Drift',
            'description': 'Drift After Dipole #1',
            'M_4x4': Md,
            'D_4x1': np.zeros(4, dtype=np.float64),
            's_elem': self.L_drift
        })

        # 8) Quad (Focusing)
        M_qf = quad_transfer_matrix_4x4(self.k_f, self.L_quad, self.use_thin_lens)
        elements.append({
            'element_type': 'Quad',
            'description': 'Focusing Quad',
            'M_4x4': M_qf,
            'D_4x1': np.zeros(4, dtype=np.float64),
            's_elem': self.L_quad
        })

        return elements


class LatticeReference:
    def __init__(self, simulator):
        """
        Initialize LatticeReference from SynchrotronSimulator.

        Parameters:
            simulator (SynchrotronSimulator): The simulator instance to extract parameters from.
        """
        self.design_radius = simulator.design_radius
        self.circumference = simulator.circumference
        self.n_FODO = simulator.n_FODO
        self.cell_length = simulator.cell_length
        self.total_FODO_length = simulator.total_FODO_length
        self.n_Dipoles = simulator.n_Dipoles
        self.f = simulator.f
        self.k_nominal = simulator.k_nominal
        self.L_quad = simulator.L_quad
        self.L_straight = simulator.L_straight
        self.L_drift = simulator.L_drift
        self.L_dipole = simulator.L_dipole
        self.theta_dipole = simulator.theta_dipole
        self.n_turns = simulator.n_turns
        self.num_particles = simulator.num_particles
        self.B = simulator.B
        self.B_rho = simulator.B_rho
        self.Qx = simulator.Qx
        self.Qy = simulator.Qy
        self.turns_full_oscillation_x = simulator.turns_full_oscillation_x
        self.turns_full_oscillation_y = simulator.turns_full_oscillation_y
        # Quadrupole Errors
        self.quad_errors = simulator.quad_errors

        # Lattice elements
        self.lattice_elements_positions = simulator.lattice_elements_positions
        self.lattice_elements_description = simulator.lattice_elements_description

        # BPM positions
        self.bpm_positions = simulator.bpm_positions

    def describe(self):
        """Print the lattice reference information."""
        print()
        print("Lattice Reference:")
        print(f"Design Radius: {self.design_radius} meters")
        print(f"Lattice Circumference: {self.circumference} meters")
        print(f"Number of FODO Cells: {self.n_FODO}")
        print(f"Total Length per FODO Cell: {self.cell_length}")
        print(f"Total Length of All FODO Cells: {self.total_FODO_length:.5f} meters")
        print(f"Number of Dipoles: {self.n_Dipoles}")
        print(f"Quadrupole Focal Length (f): {self.f} meters")
        print(f"Focusing Index {self.k_nominal}")
        print(f"Quadrupole Length (L_quad): {self.L_quad} meters")
        print(f"Straight Section Length (L_straight): {self.L_straight} meters")
        print(f"Drift Section Length: {self.L_drift} meters")
        print(f"Dipole Length per Dipole: {self.L_dipole:.5f} meters")
        print(f"Dipole Bending Angle per Dipole: {np.degrees(self.theta_dipole):.5f} degrees")
        print(f"Number of Turns: {self.n_turns}")
        print(f"Number of Particles: {self.num_particles}")
        print(f"Magnetic Field (B): {self.B:.5f} Tesla")
        print(f"Magnetic Field Rigidity (B_rho): {self.B_rho:.5f} Tesla/meters")
        print(f"Horizontal Tune (Qx): {self.Qx:.6f}")
        print(f"Vertical Tune (Qy): {self.Qy:.6f}")
        print(f"Number of turns for Full Oscillation (X): {self.turns_full_oscillation_x}")
        print(f"Number of turns for Full Oscillation (Y): {self.turns_full_oscillation_y}")

        print()
        print("Error Configuration:")
        if self.quad_errors:
            for idx, error in enumerate(self.quad_errors):
                print(f"  Quadrupole Error {idx+1}: FODO Cell {error.FODO_index}, {error.quad_type.capitalize()}, Plane = {error.plane.capitalize()}, Quad, delta = {error.delta} m")
        else:
            print("  Quadrupole Error: None")
            
        if self.quad_tilt_errors:
            for idx, error in enumerate(self.quad_tilt_errors):
                print(f"  Quadrupole Tilt Error {idx+1}: FODO Cell {error.FODO_index}, {error.quad_type.capitalize()}, Tilt Angle = {error.tilt_angle} rad")    
        else:
            print("  Quadrupole Tilt Error: None")
        
        if self.dipole_tilt_errors:
            for idx, error in enumerate(self.dipole_tilt_errors):
                print(f"  Dipole Tilt Error {idx+1}: FODO Cell {error.FODO_index}, Dipole {error.dipole_index}, Tilt Angle = {error.tilt_angle} rad")
        else:
            print("  Dipole Tilt Error: None")


class SynchrotronSimulator:
    def __init__(self, design_radius, L_quad, L_straight, p, q, n_turns, total_dipole_bending_angle,
                 num_particles=1, n_FODO=None, L_dipole=None, n_Dipoles=None,
                 G=None, f=None, use_thin_lens=False, mag_field_range=(0.5, 2.0), dipole_length_range=(0.5, 5.0),
                 horizontal_tune_range=(0.2, 0.8), vertical_tune_range=(0.2, 0.8),
                 use_gpu=False, verbose=True, figs_save_dir='figs'):
        """
        Initialize the Synchrotron Simulator.

        Parameters:
            design_radius (float): Design bending radius (meters).
            f (float): Quadrupole focal length (meters).
            L_quad (float): Length of quadrupole magnets (meters).
            L_straight (float): Length of straight sections (meters).
            p (float): Particle momentum (kg m/s).
            q (float): Particle charge (C).
            n_turns (int): Number of revolutions to simulate.
            total_dipole_bending_angle (float): radians
            num_particles (int): Number of particles to simulate.
            n_FODO (int): Number of FODO cells.
            L_dipole (float): Length of each dipole magnet (meters).
            n_Dipoles (int): Total number of dipoles in the ring.
            mag_field_range (tuple): (min_B, max_B) in Tesla.
            dipole_length_range (tuple): (min_Ld, max_Ld) in meters.
            horizontal_tune_range (tuple): (min_Qx, max_Qx).
            vertical_tune_range (tuple): (min_Qy, max_Qy)
            use_gpu (bool): If True, perform simulation on GPU using Numba. Defaults to False.
            verbose (bool): If True, print progress and information about the lattice and the simulation
        """
        # Design parameters
        self.design_radius = design_radius
        self.L_quad = L_quad
        self.L_straight = L_straight
        self.total_dipole_bending_angle = total_dipole_bending_angle  # In radians
        self.theta_dipole = None  # To be calculated
        self.use_thin_lens = use_thin_lens

        # Particle parameters
        # Conversion factor from GeV/c to kg·m/s
        GEV_C_TO_KG_M_S = 5.344286e-19  # 1 GeV/c ≈ 5.344286 x 10⁻¹⁹ kg·m/s
        self.GeV_c_to_Kg_m_s = GEV_C_TO_KG_M_S

        self.p = p
        self.q = q
        self.rho = self.design_radius  # Assuming dipoles bend along the design radius
        self.p_over_q = self.p / self.q  # # kg·m/(C·s) Momentum per charge (m)

        # Compute magnetic rigidity
        # self.B_rho = self.p_over_q  # (# Tesla·meters) B_rho = p/q / rho (or B * rho)

        # Compute the magnetic field B
        # self.B = self.B_rho / self.rho  # Tesla
        self.B = self.p / (self.q * self.design_radius)  # Tesla
        self.B_rho = self.B * self.design_radius  # Verify B_rho

        self.R_dipole_curv_radius = self.p / (self.q * self.B)

        # effective focusing power based on the quadrupole gradient and magnetic rigidity
        self.G = G
        self.k_nominal = self.G / self.B_rho  # (m⁻²)
        self.k_f_nominal = self.k_nominal  # +k
        self.k_d_nominal = -self.k_nominal  # -k
        self.f = 1 / self.k_f_nominal

        if self.rho == 0:
            raise ValueError("Bending radius rho cannot be zero.")

        assert np.isclose(self.B * self.rho, self.p_over_q, atol=1e-12), "Magnetic rigidity mismatch!"

        # Simulation parameters
        self.n_turns = n_turns
        self.num_particles = num_particles
        self.circumference = None
        self.cell_length = None

        # Lattice and transfer matrices
        self.n_FODO = n_FODO
        self.L_dipole = L_dipole
        self.n_Dipoles = n_Dipoles
        self.total_FODO_length = None
        self.L_drift = None
        
        # self.Mx_lattice_cell = []
        # self.My_lattice_cell = []
        # self.Dx_lattice_cell = []
        # self.Dy_lattice_cell = []
        # self.theta_lattice_cell = []
        
        # We'll store the final ring as lists of dicts with 'M_4x4', 's_elem', etc.
        self.M_lattice_4x4 = [] # list of 4x4 for each element
        self.D_lattice_4x1   = [] # list of 4×1 for each element
        self.lattice_positions = []
        self.bpm_positions = []
        self.len_per_cell_list = []
        
        # self.len_per_rev = None
        # self.len_per_cell = None
        
        # Initialize len_per_cell_list as an empty list
        self.len_per_cell_list = []

        # Error configuration
        self.include_quad_error = False
        self.quad_errors = []  # List of QuadrupoleMisAlignError instances

        self.quadrupole_tilt_errors = [] # List of QuadrupoleTiltError instances
        self.dipole_tilt_errors = [] # List of DipoleTiltError instances


        # Tune and phase advances
        self.mu_x = None
        self.Qx = None
        self.mu_y = None
        self.Qy = None
        self.turns_full_oscillation_x = None
        self.turns_full_oscillation_y = None

        # Particle states and positions
        self.particles_states_x = []
        self.particles_states_y = []
        self.particles_avg_x_positions = []
        self.particles_avg_y_positions = []
        self.x_global_all = []
        self.y_global_all = []

        # Lattice elements positions
        self.lattice_elements_positions = []
        self.lattice_elements_description = []

        # BPM Readings
        self.bpm_readings = {
            'x': None,
            'y': None,
            'xp': None,
            'yp': None
        }

        # Beam Size Parameters
        self.epsilon_horizontal = None  # Beam width in X (meters)
        self.epsilon_vertical = None    # Beam height in Y (meters)

        # New Constraints
        self.min_B, self.max_B = mag_field_range
        self.min_Ld, self.max_Ld = dipole_length_range
        self.min_Qx, self.max_Qx = horizontal_tune_range
        self.min_Qy, self.max_Qy = vertical_tune_range

        # GPU flag
        self.use_gpu = use_gpu  # Flag to determine computation device
        # Print backend info message
        backend = ''
        if self.use_gpu:
            backend='GPU'
        else:
            backend='cpu'
        self.backend_uasge_msg = f"[Info] Using `{backend}` backend for simulation."
        
        # Usefull for debugging
        self.verbose = verbose
        
        self.figs_save_dir = figs_save_dir
        if not os.path.exists(f"{self.figs_save_dir}"):
            os.makedirs(f"{self.figs_save_dir}")
            
        # Initialize the simulation
        self.bpm_positions = []  # Initialize BPM positions list
        self.design_synchrotron()
        self.build_lattice()
        self.verify_transfer_matrices()
        self.compute_tunes()

    def get_lattice_reference(self):
        """Return a LatticeReference instance."""
        return LatticeReference(self)

    def describe(self):
        """Print the synchrotron structure and configuration."""
        print()
        print("Synchrotron Configuration:")
        print(f"Design Radius: {self.design_radius} meters")
        print(f"Lattice Circumference: {self.circumference} meters")
        print(f"Number of FODO Cells: {self.n_FODO}")
        print(f"Total Length per FODO Cell: {self.cell_length}")
        print(f"Total Length of All FODO Cells: {self.total_FODO_length:.5f} meters")
        print(f"Number of Dipoles: {self.n_Dipoles}")
        print(f"Quadrupole Focal Length (f): {self.f} meters")
        print(f"Quadrupole Length (L_quad): {self.L_quad} meters")
        print(f"Dipole Curviture Radius (not used) = {self.R_dipole_curv_radius}")
        print(f"Focusing Index {self.k_nominal}")
        print(f"Straight Section Length (L_straight): {self.L_straight} meters")
        print(f"Drift Section Length: {self.L_drift} meters")
        print(f"Dipole Length per Dipole: {self.L_dipole:.5f} meters")
        print(f"Dipole Bending Angle per Dipole: {np.degrees(self.theta_dipole):.5f} degrees")
        print(f"Number of Turns: {self.n_turns}")
        print(f"Number of Particles: {self.num_particles}")
        print(f"Magnetic Field (B): {self.B:.5f} Tesla")
        print(f"Magnetic Field Rigidity (B_rho): {self.B_rho:.5f} Tesla/meters")
        print(f"Horizontal Tune (Qx): {self.Qx:.6f}")
        print(f"Vertical Tune (Qy): {self.Qy:.6f}")
        print(f"Number of turns for Full Oscillation (X): {self.turns_full_oscillation_x}")
        print(f"Number of turns for Full Oscillation (Y): {self.turns_full_oscillation_y}")

        print("Error Configuration:")
        if self.quad_errors:
            for idx, error in enumerate(self.quad_errors):
                print(f"  Quadrupole Error {idx+1}: FODO Cell {error.FODO_index}, {error.quad_type.capitalize()}, Plane = {error.plane.capitalize()}, Quad, delta = {error.delta} m")
        else:
            print("  Quadrupole Error: None")
            
        if self.quadrupole_tilt_errors:
            for idx, error in enumerate(self.quadrupole_tilt_errors):
                print(f"  Quadrupole Tilt Error {idx+1}: FODO Cell {error.FODO_index}, {error.quad_type.capitalize()}, Tilt Angle = {error.tilt_angle} rad")    
        else:
            print("  Quadrupole Tilt Error: None")
        
        if self.dipole_tilt_errors:
            for idx, error in enumerate(self.dipole_tilt_errors):
                print(f"  Dipole Tilt Error {idx+1}: FODO Cell {error.FODO_index}, Dipole {error.dipole_index}, Tilt Angle = {error.tilt_angle} rad")
        else:
            print("  Dipole Tilt Error: None")


        print("\nSynchrotron Structure:")
        print("Elements in one FODO cell:")
        for idx, elem in enumerate(self.lattice_elements_description):
            print(f"  Index {idx}: {elem}")

        # Now print the elements positions in each FODO cell
        print("\nElements positions in each FODO cell (positions in meters):")
        elements_per_cell = {}
        for elem in self.lattice_elements_positions:
            cell_idx = elem['cell_index']
            if cell_idx not in elements_per_cell:
                elements_per_cell[cell_idx] = []
            elements_per_cell[cell_idx].append(elem)
        # Now print elements per cell
        for cell_idx in sorted(elements_per_cell.keys()):
            print(f"\nFODO Cell {cell_idx}:")
            for elem_idx, elem in enumerate(elements_per_cell[cell_idx]):
                elem_type = elem['element_type']
                description = elem['description']
                start_s = elem['start_s']
                end_s = elem['end_s']
                length = end_s - start_s
                print(f"  Element {elem_idx}: {description}, Type: {elem_type}, Start s: {start_s:.5f} m, End s: {end_s:.5f} m, Length: {length:.5f} m")

    def drift(self, L):
        """Transfer matrix for a drift space of length L."""
        return np.array([[1, L],
                         [0, 1]])

    def design_synchrotron(self):
        """Designs the synchrotron given the design parameters and constraints."""
        # Calculate the circumference
        circumference = 2 * np.pi * self.design_radius
        self.circumference = circumference

        # Use the provided total dipole bending angle
        total_dipole_bending_angle = self.total_dipole_bending_angle

        # Calculate the dipole bending angle per dipole
        self.theta_dipole = total_dipole_bending_angle / self.n_Dipoles

        # Calculate the length of each dipole magnet
        self.L_dipole = self.design_radius * self.theta_dipole

        # Total dipole length
        total_dipole_length = self.n_Dipoles * self.L_dipole

        # Total straight length
        total_straight_length = circumference - total_dipole_length

        if total_straight_length <= 0:
            raise ValueError("Total straight length is negative or zero. Adjust the total dipole bending angle or design radius.")

        # Number of straight sections (assumed equal to n_FODO)
        num_straight_sections = self.n_FODO
        length_per_straight_section = total_straight_length / num_straight_sections

        # Distribute length among drifts and quadrupoles
        total_quad_length_per_cell = 2 * self.L_quad  # One focusing and one defocusing quad per cell
        total_drift_length_per_cell = length_per_straight_section - total_quad_length_per_cell

        if total_drift_length_per_cell <= 0:
            raise ValueError("Total drift length per cell is negative or zero. Adjust quadrupole length or total dipole bending angle.")

        # Number of drifts per cell (assumed 4)
        L_drift = total_drift_length_per_cell / 4

        if L_drift <= 0:
            raise ValueError(f"Calculated drift length (L_drift = {L_drift:.5f} m) is negative or zero. Check your parameters.")

        # Update drift lengths
        self.L_drift = L_drift
        self.L_straight = 2 * (L_drift + self.L_quad)

        # Total length per FODO cell
        cell_length = 2 * self.L_dipole + length_per_straight_section
        self.cell_length = cell_length
        self.total_FODO_length = self.n_FODO * cell_length

        # Check that the total length matches the circumference
        length_difference = abs(self.circumference - self.total_FODO_length)
        if length_difference > 1e-6:
            raise ValueError(f"Total length of FODO cells ({self.total_FODO_length:.6f} m) does not match circumference ({self.circumference:.6f} m).")

        # Calculate magnetic field B using B = p / (q * rho)
        self.B = self.p / (self.q * self.design_radius)  # Tesla
        self.B_rho = self.B * self.design_radius  # Tesla·meters

        # Check magnetic field constraints
        if not (self.min_B <= self.B <= self.max_B):
            raise ValueError(f"Magnetic field (B = {self.B:.5f} T) out of bounds [{self.min_B} T, {self.max_B} T]. "
                            f"Adjust design radius or number of dipoles.")

        # Check dipole length constraints
        if not (self.min_Ld <= self.L_dipole <= self.max_Ld):
            raise ValueError(f"Dipole length (L_dipole = {self.L_dipole:.5f} m) out of bounds [{self.min_Ld} m, {self.max_Ld} m]. "
                            f"Adjust design radius or number of dipoles.")


    def set_quad_error(self, FODO_index, quad_type, delta, plane):
        """
        Set an error in a quadrupole.

        Parameters:
            FODO_index (int): Index of the FODO cell where the error is introduced (0-based).
            quad_type (str): 'focusing' or 'defocusing'.
            delta (float): Displacement error (meters).
            plane (str): 'horizontal' or 'vertical' indicating which plane to apply the error.

        Raises:
            ValueError: If delta exceeds the stability limit or if plane is invalid.
        """
        # Create a QuadrupoleMisAlignError instance
        quad_error = QuadrupoleMisAlignError(FODO_index, quad_type, delta, plane)
        self.include_quad_error = True
        self.quad_errors.append(quad_error)

    def set_dipole_tilt_error(self, FODO_index, dipole_index, tilt_angle):
        """
        Set a tilt error in a dipole, identified by 'element_index'
        (or however you keep track of dipole indices).
        """
        self.dipole_tilt_errors.append(DipoleTiltError(FODO_index, dipole_index, tilt_angle))

    def set_quadrupole_tilt_error(self, FODO_index, quad_type, tilt_angle):
        """
        Set a tilt error in a quadrupole, identified by 'FODO_index'
        (or global element index).
        """
        self.quadrupole_tilt_errors.append(QuadrupoleTiltError(FODO_index, quad_type, tilt_angle))


    def build_lattice(self):
        """
        Build the 4x4 lattice for the ring.

        This method:
        - Resets self.M_lattice_4x4 and related arrays.
        - Creates one FODO cell (via something like FODOCell4D(...).get_elements_4x4()).
        - Replicates it n_FODO times.
        - For each dipole or quad, inserts:
            1) A tilt error element if specified in self.dipole_tilt_errors or self.quadrupole_tilt_errors.
            2) The old 'Quad' misalignment error (displacement-based) as an extra 4x4 "kick."
        - Records element positions in self.lattice_elements_positions.
        - Updates self.len_per_cell_list.
        """

        # 1) Reset the lattice-related structures
        self.M_lattice_4x4 = []  # list of 4x4 numpy arrays for each element
        self.D_lattice_4x1 = []  # list of 4x4 numpy arrays for each element
        self.lattice_elements_positions = []
        self.len_per_cell_list = []
        cumulative_s = 0.0  # to track longitudinal position s

        # Quadrupole focusing strengths
        if self.G is not None:
            # Use G to calculate k
            k_f_nominal = self.G / self.B_rho
            k_d_nominal = -k_f_nominal
            # Optionally, calculate f for reference
            self.f = 1 / k_f_nominal
        else:
            # Use f to calculate k
            k_f_nominal = float(1 / self.f)
            k_d_nominal = float(-1 / self.f)

        # Adjusted drift lengths
        L_drift = self.L_drift
        # create the FODO cell object
        fodo_cell = FODOCell4D(
            L_quad=self.L_quad,
            L_drift=L_drift,
            L_dipole=self.L_dipole,
            theta_dipole=self.theta_dipole,
            k_f=k_f_nominal,
            k_d=k_d_nominal,
            rho=self.rho,
            k_nominal=self.k_nominal,
            use_thin_lens=self.use_thin_lens
        )
                
        cell_template = fodo_cell.get_elements_4x4()  # returns 8 elements if n_FODO=8

        # Save the description of elements in one FODO cell
        self.lattice_elements_description = [elem['description'] for elem in cell_template]

        global_elem_index = 0  # If you want to keep track of a global index

        # 3) Replicate n_FODO times
        for cell_index in range(self.n_FODO):
            cell_elements_count = 0

            # Loop over the template elements in one cell
            for elem in cell_template:
                elem_type = elem['element_type']
                description = elem['description']
                M_4x4 = elem['M_4x4']
                D_4x1 = elem['D_4x1']
                s_elem = elem['s_elem']

                        # OLD TILT WAY
                        # # 3a) Insert the main element's 4x4
                        # self.M_lattice_4x4.append(M_4x4)
                        # self.D_lattice_4x1.append(D_4x1)

                # Record its start/end positions
                start_s = cumulative_s
                end_s   = cumulative_s + s_elem
                cumulative_s = end_s

                self.lattice_elements_positions.append({
                    'cell_index': cell_index,
                    'element_type': elem_type,
                    'description': description,
                    'start_s': start_s,
                    'end_s': end_s
                })

                global_elem_index += 1
                cell_elements_count += 1

                # 3b) If it's a dipole, check for a DIPOLE tilt error
                if elem_type == 'Dipole':
                    # We can parse which dipole # from the description => e.g. "Dipole #0" => dipole_number=0
                    dipole_number = 0 if '#0' in description else 1

                    # see if there's a matching tilt
                    for tilt_err in self.dipole_tilt_errors:
                        if (tilt_err.FODO_index == cell_index
                            and tilt_err.dipole_index == dipole_number):

                                    # OLD TILT WAY
                                    # Insert a zero-length tilt element
                                    # tilt_elem = make_tilt_element_4x4(
                                    #     tilt_err.tilt_angle,
                                    #     description=f"Dipole Tilt after {description} in cell {cell_index}"
                                    # )
                                    # self.M_lattice_4x4.append(tilt_elem['M_4x4'])
                                    # D_kick = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
                                    # self.D_lattice_4x1.append(D_kick)
                                    
                                    # # positions for the tilt element => s_elem=0 => same start/end
                                    # self.lattice_elements_positions.append({
                                    #     'cell_index': cell_index,
                                    #     'element_type': 'DipoleTiltError',
                                    #     'description': tilt_elem['description'],
                                    #     'start_s': cumulative_s,
                                    #     'end_s': cumulative_s
                                    # })
                                    # cell_elements_count += 1
                            
                            # NEW TILT, correct way
                            tilt_angle = tilt_err.tilt_angle
                            M_4x4 = tilt_4x4(tilt_angle) @ M_4x4
                            updated_description  = elem['description'] + f" + DipoleTilt({tilt_angle:.4g} rad)"
                            self.lattice_elements_positions[-1]['description'] = updated_description

                            if self.verbose:
                                print(f"Inserted DipoleTiltError @ cell={cell_index}, dipole={dipole_number}, angle={tilt_err.tilt_angle}")

                # 3c) If it's a quad, check for QUAD tilt error & old misalignment
                elif elem_type == 'Quad':
                    # figure out focusing vs defocusing
                    isFocusing = ('Focusing' in description)
                    quad_type_str = 'focusing' if isFocusing else 'defocusing'

                    # -- Quadrupole Tilt Error --
                    for tilt_err in self.quadrupole_tilt_errors:
                        if (tilt_err.FODO_index == cell_index
                            and tilt_err.quad_type == quad_type_str):

                                    # OLD TILT WAY
                                    # Insert a zero-length tilt element
                                    # tilt_elem = make_tilt_element_4x4(
                                    #     tilt_err.tilt_angle,
                                    #     description=f"Quad Tilt after {description} in cell {cell_index}"
                                    # )
                                    # self.M_lattice_4x4.append(tilt_elem['M_4x4'])
                                    # D_kick = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
                                    # self.D_lattice_4x1.append(D_kick)

                                    # self.lattice_elements_positions.append({
                                    #     'cell_index': cell_index,
                                    #     'element_type': 'QuadTiltError',
                                    #     'description': tilt_elem['description'],
                                    #     'start_s': cumulative_s,
                                    #     'end_s': cumulative_s
                                    # })
                                    # cell_elements_count += 1

                            
                            # NEW TILT, correct way
                            # apply sandwiched tilt
                            tilt_angle = tilt_err.tilt_angle
                            Rp = tilt_4x4(+tilt_angle)
                            Rm = tilt_4x4(-tilt_angle)
                            # Sandwich the original quad matrix
                            M_4x4 = Rm @ M_4x4 @ Rp
                            updated_description  = elem['description'] + f" + QuadTilt({tilt_angle:.4g} rad)"
                            self.lattice_elements_positions[-1]['description'] = updated_description
                            
                            if self.verbose:
                                print(f"\tInserted QuadTiltError @ cell={cell_index}, type={quad_type_str}, angle={tilt_err.tilt_angle}")
                                

                    # -- The OLD "Quad" misalignment error (displacement-based) --
                    # For each quad in cell, we used to add a dipole-kick block if there's a misalignment
                    # We replicate that in 4x4 form. 
                    # theta_kick = k_nominal * delta * L_quad (sign depends on plane).
                    # define a small function to build that 4x4 "kick."

                    for quad_err in self.quad_errors:
                        if (quad_err.FODO_index == cell_index
                            and quad_err.quad_type == quad_type_str):
                            # e.g. plane='vertical' => we do a vertical angle
                            plane = quad_err.plane
                            delta = quad_err.delta
                            L_q   = self.L_quad
                            k_nominal = k_f_nominal if (quad_type_str=='focusing') else k_d_nominal

                            # compute the small angle
                            theta_kick = k_nominal * delta * L_q
                            # sign => if plane='vertical', means a rotation in y-plane

                            # Build a 4x4 that adds a small angle to x' or y'
                            M_kick = np.identity(4, dtype=np.float64)
                            # horizontal misalignment => x' -> x' + theta_kick
                            # i.e. M_kick[1,0] = +theta_kick
                            # vertical => y' -> y' + theta_kick => M_kick[3,2] = +theta_kick
                            if plane=='horizontal':
                                D_kick = np.array([0.0, theta_kick, 0.0, 0.0], dtype=np.float64)  # [x, x_p, y, y_p]
                            elif plane=='vertical':
                                D_kick = np.array([0.0, 0.0, 0.0, theta_kick], dtype=np.float64)  # [x, x_p, y, y_p]

                            # Insert it as a zero-length element
                            self.M_lattice_4x4.append(M_kick)
                            self.D_lattice_4x1.append(D_kick)

                            self.lattice_elements_positions.append({
                                'cell_index': cell_index,
                                'element_type': 'QuadMisalignmentKick',
                                'description': f"Quad Displacement Kick after {description}, plane={plane}, delta={delta}",
                                'start_s': cumulative_s,
                                'end_s': cumulative_s
                            })
                            cell_elements_count += 1

                            if self.verbose:
                                print(f"Applied old Quad misalignment at cell={cell_index}, quad={quad_type_str}, plane={plane}, delta={delta}, theta_kick={theta_kick}")

                # (NEW TILT, correct way) Append the (possibly tilted) M_4x4:
                self.M_lattice_4x4.append(M_4x4)
                self.D_lattice_4x1.append(D_4x1)

            # End of cell
            self.len_per_cell_list.append(cell_elements_count)

        # 4) After building all cells, set self.len_per_rev
        self.len_per_rev = sum(self.len_per_cell_list)

        if self.verbose:
            print(f"build_lattice() completed. Total elements per rev = {self.len_per_rev}")


    def compute_twiss_parameters(self):
        """
        Method 1:
        Compute Twiss parameters (alpha, beta, gamma) for both planes, 
        by extracting the 2x2 diagonal blocks from the final 4x4 one-turn matrix.

        Steps:
        1) Build M_ring (4x4) by multiplying all 4x4 element matrices in the lattice.
        2) Extract Mx = M_ring[:2,:2], My = M_ring[2:,2:].
        3) Compute phase advance mu_x, mu_y from the trace of Mx, My.
        4) Compute (alpha_x, beta_x, gamma_x) and (alpha_y, beta_y, gamma_y).
        5) Return them for external use or logging.

        NOTE:
        - This approach discards cross-plane coupling, so it is valid only if
            the coupling is negligible or if you simply want approximate Twiss 
            in each plane.
        - For a fully coupled 4D approach, you'd do a normal-form analysis 
            rather than extracting sub-blocks.

        Returns:
            alpha_x, beta_x, gamma_x, alpha_y, beta_y, gamma_y
        """
        if not getattr(self, 'M_lattice_4x4', None):
            raise ValueError("4x4 lattice matrices (M_lattice_4x4) are not initialized. "
                            "Please build the lattice before computing Twiss parameters.")

        # 1) Build the one-turn 4x4
        M_ring_4x4 = np.identity(4)
        for M_elem in self.M_lattice_4x4:
            M_ring_4x4 = M_elem @ M_ring_4x4

        # 2) Extract horizontal sub-block Mx (2x2), vertical sub-block My (2x2)
        Mx = M_ring_4x4[0:2, 0:2]
        My = M_ring_4x4[2:4, 2:4]

        # Helper function to compute (alpha, beta, gamma) from a 2x2 one-turn matrix
        def twiss_from_2x2(M2):
            m11, m12 = M2[0,0], M2[0,1]
            m21, m22 = M2[1,0], M2[1,1]
            trace   = m11 + m22
            cos_mu  = trace / 2.0

            # Clip cos_mu to [-1, 1] for safety
            if cos_mu < -1.0:
                cos_mu = -1.0
            elif cos_mu > 1.0:
                cos_mu = 1.0

            mu   = np.arccos(cos_mu)
            # If sin(mu) is near zero => unstable or pi, handle carefully
            sin_mu = np.sin(mu)
            if abs(sin_mu) < 1e-12:
                # fallback
                beta  = np.nan
                alpha = np.nan
                gamma = np.nan
            else:
                beta  = m12 / sin_mu
                alpha = (m11 - m22) / (2.0*sin_mu)
                gamma = (1.0 + alpha**2) / beta

            return alpha, beta, gamma

        # 3) Compute horizontal Twiss
        alpha_x, beta_x, gamma_x = twiss_from_2x2(Mx)
        # 4) Compute vertical Twiss
        alpha_y, beta_y, gamma_y = twiss_from_2x2(My)

        return alpha_x, beta_x, gamma_x, alpha_y, beta_y, gamma_y


    def generate_initial_states_from_twiss(self, alpha_x, beta_x, epsilon_x, alpha_y, beta_y, epsilon_y, num_particles):
        """
        Method 2:
        Generate initial particle states (x, x', y, y') that comply with given Twiss parameters and emittances.
        Particles are assumed to be drawn from a 2D Gaussian distribution in each plane defined by:
        
        Sigma_x = epsilon_x * [[ beta_x,      -alpha_x ],
                            [ -alpha_x, (1+alpha_x^2)/beta_x ]]
        
        Sigma_y = epsilon_y * [[ beta_y,      -alpha_y ],
                            [ -alpha_y, (1+alpha_y^2)/beta_y ]]

        This generates phase space distributions consistent with the specified Twiss parameters and emittances.

        Parameters:
            alpha_x, beta_x, epsilon_x: Twiss and emittance for horizontal plane.
            alpha_y, beta_y, epsilon_y: Twiss and emittance for vertical plane.
            num_particles (int): Number of particles.

        Returns:
            initial_states (np.ndarray): shape (num_particles,4) containing [x, x', y, y'].
        """
        # Compute gamma parameters
        gamma_x = (1 + alpha_x**2) / beta_x
        gamma_y = (1 + alpha_y**2) / beta_y

        epsilon_x = 1e-6  # 1 micrometer·rad emittance for horizontal plane
        epsilon_y = 1e-6  # # 1 micrometer·rad emittance for vertical plane

        # Construct covariance matrices
        Sigma_x = epsilon_x * np.array([[beta_x,      -alpha_x],
                                        [-alpha_x,    gamma_x]])
        Sigma_y = epsilon_y * np.array([[beta_y,      -alpha_y],
                                        [-alpha_y,    gamma_y]])

        # Generate horizontal coordinates (x, x')
        X_hor = np.random.multivariate_normal(mean=[0,0], cov=Sigma_x, size=num_particles)
        # Generate vertical coordinates (y, y')
        X_ver = np.random.multivariate_normal(mean=[0,0], cov=Sigma_y, size=num_particles)

        # Combine into initial states array
        initial_states = np.column_stack([X_hor[:,0], X_hor[:,1], X_ver[:,0], X_ver[:,1]])
        return initial_states

    def verify_transfer_matrices(self):
        """
        Verify that all 4x4 transfer matrices are symplectic and contain finite values.

        A 4D matrix M is symplectic if M^T J M = J, where 
        J = [[0, 1, 0, 0],
            [-1,0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, -1,0]].
        We check if || M^T J M - J || is small.

        Also checks if each matrix has only finite (non-NaN, non-inf) entries.

        Prints warnings if a matrix fails these checks.
        """
        # print("verify_transfer_matrices..")

        if not getattr(self, 'M_lattice_4x4', None):
            print("No 4x4 lattice matrices to verify. Possibly not built yet.")
            return

        # Define the symplectic form J in 4D
        J = np.array([[0, 1, 0, 0],
                    [-1,0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, -1,0]], dtype=np.float64)

        for idx, M in enumerate(self.M_lattice_4x4):
            # 1) Check if all elements are finite
            if not np.all(np.isfinite(M)):
                print(f"Warning: 4x4 transfer matrix M_lattice_4x4[{idx}] contains non-finite values.")
                continue

            # 2) Check symplectic condition
            # We compute M^T @ J @ M
            M_tJ  = M.T @ J
            test  = M_tJ @ M
            diff  = test - J
            norm  = np.linalg.norm(diff)  # e.g. Frobenius norm

            if norm > 1e-6:
                print(f"Warning: 4x4 transfer matrix M_lattice_4x4[{idx}] not symplectic (norm diff={norm:.2e}).")

            # 3) Optionally also check det(M). For a perfect symplectic in 4D, det(M)=1, but 
            #    that alone doesn't guarantee no coupling errors. Still let's do it:
            detM = np.linalg.det(M)
            if abs(detM - 1.0) > 1e-6:
                print(f"Warning: M_lattice_4x4[{idx}] has determinant={detM:.6f}, not ~1.0 => possibly non-symplectic.")

    @classmethod
    def find_feasible_lattices(cls, base_configurations, common_parameters, verbose=True):
        feasible_configs = []
        rejection_reasons = {}

        for base_config in base_configurations:
            merged_config = {**common_parameters, **base_config}
            # Extract parameters as before...
            design_radius = merged_config['design_radius']
            L_quad = merged_config['L_quad']
            G = merged_config.get('G', None)
            f = merged_config.get('f', None)
            p = merged_config.get('p', None)
            q = merged_config.get('q', None)
            mag_field_range = merged_config.get('mag_field_range', (0.1, 2.0))
            dipole_length_range = merged_config.get('dipole_length_range', (0.2, 14.0))
            # Tune range
            horizontal_tune_range = merged_config.get('horizontal_tune_range', (0.1, 0.8))
            vertical_tune_range = merged_config.get('vertical_tune_range', (0.1, 0.8))
            # Total dipole bending angle range
            total_dipole_bending_angle_range = merged_config.get('total_dipole_bending_angle_range', (np.pi, 1.9 * np.pi))

            # Possible values for n_FODO, n_Dipoles
            if 'n_FODO' in merged_config:
                n_FODO_values = [merged_config['n_FODO']]  # Use specified n_FODO
            else:
                n_FODO_values = range(2, 21)  # Number of FODO cells

            if 'n_Dipoles' in merged_config:
                n_Dipoles_values = [merged_config['n_Dipoles']]  # Use specified n_Dipoles
            else:
                n_Dipoles_values = range(2, 41, 2)  # Number of dipoles (even numbers)

            # Define total_dipole_bending_angle values
            total_dipole_bending_angle_min, total_dipole_bending_angle_max = total_dipole_bending_angle_range
            total_dipole_bending_angle_values = np.linspace(
                total_dipole_bending_angle_min, total_dipole_bending_angle_max, 10  # Adjust number of steps as needed
            )

            for total_dipole_bending_angle in total_dipole_bending_angle_values:
                for n_FODO in n_FODO_values:
                    for n_Dipoles in n_Dipoles_values:
                        # Ensure n_Dipoles is at least 2 * n_FODO
                        if n_Dipoles < 2 * n_FODO:
                            reason = 'n_Dipoles less than 2 times n_FODO'
                            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                            continue  # Not enough dipoles to cover all FODO cells

                        # Calculate theta_dipole
                        theta_dipole = total_dipole_bending_angle / n_Dipoles

                        # Avoid unphysical theta_dipole values
                        if theta_dipole <= 0 or theta_dipole > 2 * np.pi:
                            reason = 'Invalid theta_dipole value'
                            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                            continue

                        L_dipole = theta_dipole * design_radius

                        # Check dipole length constraints
                        if not (dipole_length_range[0] <= L_dipole <= dipole_length_range[1]):
                            reason = 'Dipole length out of range'
                            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                            continue  # Dipole length out of acceptable range

                        # Calculate total dipole length
                        total_dipole_length = n_Dipoles * L_dipole

                        # Calculate circumference
                        circumference = 2 * np.pi * design_radius

                        # Calculate total straight length
                        total_straight_length = circumference - total_dipole_length

                        if total_straight_length <= 0:
                            reason = 'Total straight length is negative or zero'
                            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                            continue  # No room for straight sections

                        # Length per straight section
                        length_per_straight_section = total_straight_length / n_FODO

                        # Distribute length among drifts and quadrupoles
                        total_quad_length_per_cell = 2 * L_quad  # One focusing and one defocusing quad per cell
                        total_drift_length_per_cell = length_per_straight_section - total_quad_length_per_cell

                        if total_drift_length_per_cell <= 0:
                            reason = 'Total drift length per cell is negative or zero'
                            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                            continue  # Not enough length for drifts

                        # Number of drifts per cell (assumed 4)
                        L_drift = total_drift_length_per_cell / 4

                        if L_drift <= 0:
                            reason = 'Calculated drift length is negative or zero'
                            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                            continue  # Invalid drift length

                        # Total length per FODO cell
                        cell_length = 2 * L_dipole + length_per_straight_section
                        total_FODO_length = n_FODO * cell_length

                        length_difference = abs(circumference - total_FODO_length)

                        if length_difference > 1e-6:
                            reason = 'Total length does not match circumference'
                            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                            continue  # Lengths do not match

                        check_config = {
                            'design_radius': design_radius,
                            'L_quad': L_quad,
                            'n_FODO': n_FODO,
                            'L_straight': length_per_straight_section,
                            'L_dipole': L_dipole,
                            'n_Dipoles': n_Dipoles,
                            'L_drift': L_drift,
                            'G': G,
                            'p': p,
                            'q': q,
                            'total_dipole_bending_angle': total_dipole_bending_angle
                        }
                        print("check_config = ", check_config)

                        # Prepare the merged configuration for the simulator
                        simulator_config = merged_config.copy()
                        simulator_config['n_FODO'] = n_FODO
                        simulator_config['L_straight'] = length_per_straight_section
                        simulator_config['L_dipole'] = L_dipole
                        simulator_config['n_Dipoles'] = n_Dipoles
                        simulator_config['total_dipole_bending_angle'] = total_dipole_bending_angle

                        # Initialize the simulator
                        try:
                            simulator = cls(
                                design_radius=design_radius,
                                G=G,
                                f=f,
                                use_thin_lens=simulator_config.get('use_thin_lens', False),
                                L_quad=L_quad,
                                L_straight=length_per_straight_section,
                                p=p,
                                q=q,
                                n_turns=merged_config.get('n_turns', 100),
                                total_dipole_bending_angle=total_dipole_bending_angle,
                                num_particles=merged_config.get('num_particles', 1),
                                n_FODO=n_FODO,
                                L_dipole=L_dipole,
                                n_Dipoles=n_Dipoles,
                                mag_field_range=mag_field_range,
                                dipole_length_range=dipole_length_range,
                                horizontal_tune_range=horizontal_tune_range,
                                vertical_tune_range=vertical_tune_range,
                                verbose=verbose,  # Suppress output during lattice search,
                                figs_save_dir=merged_config.get('figs_save_dir', 'figs'),

                            )

                            # Build the lattice and compute tunes
                            simulator.build_lattice()
                            simulator.compute_tunes()

                            # Check that the total length matches the circumference
                            if abs(simulator.total_FODO_length - simulator.circumference) > 1e-6:
                                reason = 'Total length does not match circumference after building lattice'
                                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                                continue  # Total length does not match circumference

                            # Check magnetic field constraints
                            if not (simulator.min_B <= simulator.B <= simulator.max_B):
                                reason = 'Magnetic field out of bounds'
                                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                                continue

                            # Check tune constraints
                            if not (horizontal_tune_range[0] <= simulator.Qx <= horizontal_tune_range[1]):
                                reason = 'Horizontal tune out of range'
                                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                                print("\tsimulator.Qx = ", simulator.Qx)
                                continue
                            if not (vertical_tune_range[0] <= simulator.Qy <= vertical_tune_range[1]):
                                reason = 'Vertical tune out of range'
                                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                                print("\tsimulator.Qy = ", simulator.Qy)
                                continue

                            # If all checks pass, save the configuration
                            config = {
                                'design_radius': design_radius,
                                'L_quad': L_quad,
                                'n_FODO': n_FODO,
                                'L_straight': length_per_straight_section,
                                'L_dipole': L_dipole,
                                'n_Dipoles': n_Dipoles,
                                'L_drift': L_drift,
                                'Qx': simulator.Qx,
                                'Qy': simulator.Qy,
                                'B': simulator.B,
                                'B_rho': simulator.B_rho,
                                'f': simulator.f,
                                'G': G,
                                'p': p,
                                'q': q,
                                'total_dipole_bending_angle': total_dipole_bending_angle
                            }
                            feasible_configs.append(config)
                        except ValueError as e:
                            reason = f'ValueError: {str(e)}'
                            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                            continue
                        except Exception as e:
                            reason = f'Exception: {str(e)}'
                            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                            continue

        # After the loops, print out the rejection reasons
        print("\nRejection reasons and counts:")
        for reason, count in rejection_reasons.items():
            print(f"{reason}: {count}")

        return feasible_configs

    def compute_tune_4x4(self, M_4x4):
        """
        Helper to find the two normal-mode tunes from a 4x4 matrix M_4x4.
        Returns (Q1, Q2).
        """
        evals, _ = np.linalg.eig(M_4x4)
        angles = []
        for lam in evals:
            mag = np.abs(lam)
            if mag > 1e-10:
                phase = np.angle(lam)
                tune = phase/(2*np.pi)
                if tune < 0:
                    tune += 1.0
                angles.append(tune)
        angles.sort()
        
        if self.verbose:
            print("one turn matrix:\n", M_4x4)
            print("compute_tune_4x4()/ evals = ", evals)
            print("compute_tune_4x4()/ angles = ", angles)
        
        if len(angles) < 2:
            return (np.nan, np.nan)
        if angles[0] == 0 or  angles[1] == 0:
            raise Exception("Tunes are 0. Ring is unstable.")
        return (angles[0], angles[1])

    def compute_tunes(self):
        """
        Compute horizontal/vertical tunes (Qx, Qy) from the 4x4 ring matrix.
        """
        if not getattr(self, 'M_lattice_4x4', None):
            raise ValueError("Lattice 4x4 transfer matrices are not built.")

        M_ring_4x4 = np.identity(4)
        for idx, M_elem in enumerate(self.M_lattice_4x4):
            M_ring_4x4 = M_elem @ M_ring_4x4
        
        # print("One-turn 4x4 matrix =\n", M_ring_4x4)
        
        try:
            Q1, Q2 = self.compute_tune_4x4(M_ring_4x4)
            if self.verbose:
                print("compute_tune_4x4()/ ", Q1, Q2)
            self.Qx = Q1
            self.Qy = Q2
            self.mu_x = 2*np.pi*self.Qx
            self.mu_y = 2*np.pi*self.Qy
        except Exception as e:
            raise RuntimeError(f"Error computing tunes: {e}")

        if np.isnan(self.Qx) or np.isnan(self.Qy):
            raise ValueError("Computed tunes are NaN. Possibly unstable lattice.")

        # Range checks
        if not (self.min_Qx <= self.Qx <= self.max_Qx):
            raise ValueError(f"Horizontal Tune (Qx={self.Qx:.6f}) out of range [{self.min_Qx}, {self.max_Qx}].")
        if not (self.min_Qy <= self.Qy <= self.max_Qy):
            raise ValueError(f"Vertical Tune (Qy={self.Qy:.6f}) out of range [{self.min_Qy}, {self.max_Qy}].")

        if self.verbose:
            print("compute_tunes()/", self.Qx, self.Qy)

        self.turns_full_oscillation_x = self.compute_full_oscillation_turns(self.Qx)
        self.turns_full_oscillation_y = self.compute_full_oscillation_turns(self.Qy)


    def compute_full_oscillation_turns(self, Q):
        """Compute the number of turns to complete a full oscillation."""
        if np.isnan(Q):
            return np.inf
        frac = Q % 1
        if frac == 0:
            return 1
        else:
            turns_float = 1 / frac
            turns_int = int(np.round(turns_float))
            return turns_int

    def compute_closed_orbit(self):
        """
        4D closed orbit: solve (I - M_ring)*X_co=0
        """
        if not getattr(self, 'M_lattice_4x4', None):
            raise ValueError("M_lattice_4x4 lattice not built.")
        if not getattr(self, 'D_lattice_4x1', None):
            raise ValueError("D_lattice_4x1 lattice not built.")

        M_ring_4x4 = np.identity(4)
        D_ring_4x1 = np.zeros(4, dtype=np.float64)
        
        for M_elem, D_elem in zip(self.M_lattice_4x4, self.D_lattice_4x1):
            D_ring_4x1 = M_elem @ D_ring_4x1 + D_elem
            M_ring_4x4 = M_elem @ M_ring_4x4

        I_minus_M = np.identity(4) - M_ring_4x4
        try:
            X_co_4d = np.linalg.solve(I_minus_M, D_ring_4x1)
        except np.linalg.LinAlgError:
            X_co_4d = np.zeros(4)
            print("Warning: Could not compute 4D closed orbit (matrix singular).")

        # Return plane-wise
        x_co = X_co_4d[0:2]
        y_co = X_co_4d[2:4]
        return x_co, y_co

    def simulate(self, initial_states):
        """
        Simulate the particle motion given initial conditions in 4D.
        Retains BPM recording logic, final x_global_all, etc.
        """
        if self.verbose:
            print("\n Running simulation ..")

        if self.num_particles != len(initial_states):
            raise ValueError("Number of initial states must match num_particles.")

        # BPM arrays
        if self.n_FODO:
            self.bpm_readings['x']  = np.zeros((self.num_particles, self.n_turns, self.n_FODO))
            self.bpm_readings['y']  = np.zeros((self.num_particles, self.n_turns, self.n_FODO))
            self.bpm_readings['xp'] = np.zeros((self.num_particles, self.n_turns, self.n_FODO))
            self.bpm_readings['yp'] = np.zeros((self.num_particles, self.n_turns, self.n_FODO))

        x_co, y_co = self.compute_closed_orbit()
        print(f"Closed orbit computation: x_co={x_co}, y_co={y_co}")

        if self.use_gpu:
            self._simulate_gpu(initial_states, x_co, y_co)
        else:
            self._simulate_cpu(initial_states, x_co, y_co)

        self.calculate_beam_size()

        # Reconstruct states for plotting
        self.particles_states_x = []
        self.particles_states_y = []
        self.x_global_all = []
        self.y_global_all = []

        for pid in range(self.num_particles):
            states_x = []
            states_y = []
            theta_positions = []
            theta_accumulated = 0.0

            for turn in range(self.n_turns):
                for cell in range(self.n_FODO):
                    x  = self.bpm_readings['x'][pid, turn, cell]
                    xp = self.bpm_readings['xp'][pid, turn, cell]
                    y  = self.bpm_readings['y'][pid, turn, cell]
                    yp = self.bpm_readings['yp'][pid, turn, cell]

                    states_x.append([x, xp])
                    states_y.append([y, yp])

                    # approximate ring angle
                    theta_increment = self.theta_dipole * 2
                    theta_accumulated += -theta_increment
                    theta_positions.append(theta_accumulated)

            states_x = np.array(states_x)
            states_y = np.array(states_y)
            theta_positions = np.array(theta_positions)

            x_global = (self.design_radius + states_x[:,0]) * np.cos(theta_positions)
            y_global = (self.design_radius + states_y[:,0]) * np.sin(theta_positions)

            self.particles_states_x.append(states_x)
            self.particles_states_y.append(states_y)
            self.x_global_all.append(x_global)
            self.y_global_all.append(y_global)

    def _simulate_cpu(self, initial_states, x_co, y_co):
        """CPU-based simulation in 4D."""
        nb_particles = len(initial_states)
        debug_sim_rate = 0.2
        debug_prcnt = list(range(1, int((1/debug_sim_rate)) + 1))

        init_np = np.array(initial_states, dtype=np.float64)
        # add closed orbit
        for i in range(nb_particles):
            init_np[i,0] += x_co[0]
            init_np[i,1] += x_co[1]
            init_np[i,2] += y_co[0]
            init_np[i,3] += y_co[1]

        for p_idx in range(nb_particles):
            if self.verbose and (p_idx * self.n_turns) % ((nb_particles * self.n_turns) * debug_sim_rate) == 0 and debug_prcnt:
                progress_prct = int(debug_prcnt.pop(0) * debug_sim_rate * 100)
                print(f"\t {progress_prct}% ...")

            X_4d = init_np[p_idx].copy()

            for turn in range(self.n_turns):
                # elem_idx_global = 0
                for cell_index in range(self.n_FODO):
                    n_elems = self.len_per_cell_list[cell_index]
                    for elem_in_cell in range(n_elems):
                        global_elem_idx = sum(self.len_per_cell_list[:cell_index]) + elem_in_cell
                        
                        M_4x4 = self.M_lattice_4x4[global_elem_idx]
                        D_4x1 = self.D_lattice_4x1[global_elem_idx]
                        # linear inhomogeneous transform
                        X_4d = M_4x4 @ X_4d + D_4x1
                        
                        # elem_idx_global += 1

                    self.bpm_readings['x'][p_idx, turn, cell_index]  = X_4d[0]
                    self.bpm_readings['y'][p_idx, turn, cell_index]  = X_4d[2]
                    self.bpm_readings['xp'][p_idx, turn, cell_index] = X_4d[1]
                    self.bpm_readings['yp'][p_idx, turn, cell_index] = X_4d[3]

            init_np[p_idx] = X_4d


    def _simulate_gpu(self, initial_states, x_co, y_co):
        """GPU-accelerated simulation using Numba on CUDA in 4D."""
        nb_particles = len(initial_states)
        if nb_particles == 0:
            return

        # Prepare initial states + closed orbit
        init_np = np.array(initial_states, dtype=np.float64)
        for i in range(nb_particles):
            init_np[i,0] += x_co[0]
            init_np[i,1] += x_co[1]
            init_np[i,2] += y_co[0]
            init_np[i,3] += y_co[1]

        d_states = cuda.to_device(init_np)

        n_elements = len(self.M_lattice_4x4)
        M_arr_4d = np.zeros((n_elements,4,4), dtype=np.float64)
        D_arr_4d = np.zeros((n_elements,4),   dtype=np.float64)
        
        for eix in range(n_elements):
            M_arr_4d[eix] = self.M_lattice_4x4[eix]
            # copy over the inhom vector
            D_arr_4d[eix] = self.D_lattice_4x1[eix]


        d_M_arr_4d = cuda.to_device(M_arr_4d)
        d_D_arr_4d = cuda.to_device(D_arr_4d)
        
        d_len_per_cell = cuda.to_device(np.array(self.len_per_cell_list, dtype=np.int32))

        d_bpm_x  = cuda.device_array((self.num_particles, self.n_turns, self.n_FODO), dtype=np.float64)
        d_bpm_y  = cuda.device_array((self.num_particles, self.n_turns, self.n_FODO), dtype=np.float64)
        d_bpm_xp = cuda.device_array((self.num_particles, self.n_turns, self.n_FODO), dtype=np.float64)
        d_bpm_yp = cuda.device_array((self.num_particles, self.n_turns, self.n_FODO), dtype=np.float64)

        threads_per_block = 256
        blocks_per_grid = max(1, (nb_particles + threads_per_block - 1)//threads_per_block)

        @cuda.jit
        def simulate_kernel_4D(d_states, d_M_elem, d_D_elem, len_per_cell, n_FODO, n_turns, 
                            bpm_x, bpm_y, bpm_xp, bpm_yp):
            pid = cuda.grid(1)
            if pid >= d_states.shape[0]:
                return

            # local 4D
            x0 = d_states[pid,0]
            x1 = d_states[pid,1]
            x2 = d_states[pid,2]
            x3 = d_states[pid,3]

            for turn in range(n_turns):
                # elem_idx = 0
                for cell_idx in range(n_FODO):
                    n_elems = len_per_cell[cell_idx]
                    for elem_in_cell in range(n_elems):
                        # Calculate the global element index using an explicit loop
                        global_elem_idx = 0
                        for i in range(cell_idx):
                            global_elem_idx += len_per_cell[i]
                        global_elem_idx += elem_in_cell  # Add the current element within the cell
                        
                        # multiply
                        M00 = d_M_elem[global_elem_idx,0,0]
                        M01 = d_M_elem[global_elem_idx,0,1]
                        M02 = d_M_elem[global_elem_idx,0,2]
                        M03 = d_M_elem[global_elem_idx,0,3]
                        M10 = d_M_elem[global_elem_idx,1,0]
                        M11 = d_M_elem[global_elem_idx,1,1]
                        M12 = d_M_elem[global_elem_idx,1,2]
                        M13 = d_M_elem[global_elem_idx,1,3]
                        M20 = d_M_elem[global_elem_idx,2,0]
                        M21 = d_M_elem[global_elem_idx,2,1]
                        M22 = d_M_elem[global_elem_idx,2,2]
                        M23 = d_M_elem[global_elem_idx,2,3]
                        M30 = d_M_elem[global_elem_idx,3,0]
                        M31 = d_M_elem[global_elem_idx,3,1]
                        M32 = d_M_elem[global_elem_idx,3,2]
                        M33 = d_M_elem[global_elem_idx,3,3]

                        y0 = M00*x0 + M01*x1 + M02*x2 + M03*x3
                        y1 = M10*x0 + M11*x1 + M12*x2 + M13*x3
                        y2 = M20*x0 + M21*x1 + M22*x2 + M23*x3
                        y3 = M30*x0 + M31*x1 + M32*x2 + M33*x3

                        # add inhom vector
                        D0 = d_D_elem[global_elem_idx,0]
                        D1 = d_D_elem[global_elem_idx,1]
                        D2 = d_D_elem[global_elem_idx,2]
                        D3 = d_D_elem[global_elem_idx,3]

                        x0 = y0 + D0
                        x1 = y1 + D1
                        x2 = y2 + D2
                        x3 = y3 + D3
                        # elem_idx += 1

                    # BPM
                    bpm_x[pid,turn,cell_idx]  = x0
                    bpm_y[pid,turn,cell_idx]  = x2
                    bpm_xp[pid,turn,cell_idx] = x1
                    bpm_yp[pid,turn,cell_idx] = x3

            d_states[pid,0] = x0
            d_states[pid,1] = x1
            d_states[pid,2] = x2
            d_states[pid,3] = x3

        simulate_kernel_4D[blocks_per_grid, threads_per_block](
            d_states, d_M_arr_4d, d_D_arr_4d, d_len_per_cell, self.n_FODO, self.n_turns,
            d_bpm_x, d_bpm_y, d_bpm_xp, d_bpm_yp
        )
        cuda.synchronize()

        final_states_gpu = d_states.copy_to_host()
        self.bpm_readings['x']  = d_bpm_x.copy_to_host()
        self.bpm_readings['y']  = d_bpm_y.copy_to_host()
        self.bpm_readings['xp'] = d_bpm_xp.copy_to_host()
        self.bpm_readings['yp'] = d_bpm_yp.copy_to_host()

    def calculate_beam_size(self):
        """Calculate beam size (RMS width and height) from BPM readings."""
        if self.bpm_readings['x'] is None or self.bpm_readings['y'] is None:
            print("BPMs are not placed in the lattice.")
            return

        # BPM readings are 3D numpy arrays: [num_particles, n_turns, n_FODO]
        x_measurements = self.bpm_readings['x'].flatten()
        y_measurements = self.bpm_readings['y'].flatten()

        # Compute beam sizes (standard deviations)
        self.epsilon_horizontal = np.std(x_measurements)
        self.epsilon_vertical = np.std(y_measurements)

    def plot_ring(self, ax=None, plot_xlim=None, plot_ylim=None):
        """Plot the ring and elements with start of each FODO cell indicated and BPMs rotated perpendicular to the ring."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        # Prepare the design trajectory for plotting
        theta_design = np.linspace(0, 2 * np.pi, 1000)  # Negative for clockwise rotation
        x_design = self.design_radius * np.cos(theta_design)
        y_design = self.design_radius * np.sin(theta_design)

        ax.plot(x_design, y_design, 'g--', ms=15, alpha=0.9, label='Design Trajectory')

        # Plot the elements on the design trajectory using cumulative_angle
        plotted_labels = set()

        for elem in self.lattice_elements_positions:
            element_type = elem['element_type']
            description = elem['description']
            mid_theta = elem['cumulative_angle']  # Use cumulative_angle directly

            # Compute the position on the design trajectory
            x = self.design_radius * np.cos(mid_theta)
            y = self.design_radius * np.sin(mid_theta)

            if element_type == 'Drift':
                label = 'Drift' if 'Drift' not in plotted_labels else None
                ax.scatter(x, y, color='green', marker='s', s=500, alpha=0.7, label=label)
                plotted_labels.add('Drift')
            elif element_type == 'Quad':
                if 'Focusing' in description:
                    label = 'Focusing Quad' if 'Focusing Quad' not in plotted_labels else None
                    ax.scatter(x, y, color='orange', marker='^', s=500, alpha=0.7, label=label)
                    plotted_labels.add('Focusing Quad')
                else:
                    label = 'Defocusing Quad' if 'Defocusing Quad' not in plotted_labels else None
                    ax.scatter(x, y, color='orange', marker='v', s=500, alpha=0.7, label=label)
                    plotted_labels.add('Defocusing Quad')
            elif element_type == 'Dipole':
                label = 'Dipole' if 'Dipole' not in plotted_labels else None
                ax.scatter(x, y, color='blue', marker='o', s=400, alpha=0.7, label=label)
                plotted_labels.add('Dipole')
            elif element_type == 'Dipole Kick in "Quad"':
                label = 'Dipole Kick in Quad' if 'Dipole Kick in Quad' not in plotted_labels else None
                ax.scatter(x, y, color='red', marker='X', s=400, alpha=0.7, label=label)
                plotted_labels.add('Dipole Kick in Quad')
            else:
                pass

        # Plot BPM positions rotated perpendicular to the ring (pointing towards the center)
        for bpm in self.bpm_positions:
            bpm_angle = bpm['cumulative_angle']
            x = self.design_radius * np.cos(bpm_angle)
            y = self.design_radius * np.sin(bpm_angle)

            # Calculate the angle of the line (along the radius, pointing towards the center)
            line_angle = bpm_angle  # Along radius

            # Create a line segment at the BPM location
            length = self.design_radius * 0.05  # 5% of design radius

            # Adjust dx and dy to point towards the center
            dx = -length * np.cos(line_angle)
            dy = -length * np.sin(line_angle)

            x_start = x
            x_end = x + dx
            y_start = y
            y_end = y + dy

            label = 'BPM' if 'BPM' not in plotted_labels else None
            ax.plot([x_start, x_end], [y_start, y_end], color='purple', linewidth=5, alpha=0.5, label=label)
            plotted_labels.add('BPM')

        # Add text indicating the start of each FODO cell
        elements_per_cell = {}
        for elem in self.lattice_elements_positions:
            cell_idx = elem['cell_index']
            if cell_idx not in elements_per_cell:
                elements_per_cell[cell_idx] = []
            elements_per_cell[cell_idx].append(elem)

        for cell_idx in elements_per_cell:
            cell_elements = elements_per_cell[cell_idx]
            start_elem = cell_elements[0]
            start_angle = start_elem['cumulative_angle'] % (2 * np.pi)

            # Position for text slightly outside the ring
            text_radius = self.design_radius * 1.05  # Slightly outside the ring
            x_text = text_radius * np.cos(start_angle)
            y_text = text_radius * np.sin(start_angle)

            ax.text(x_text, y_text, f'Start of Cell {cell_idx}', color='black', fontsize=8, ha='center', va='center', clip_on=True)

        # Set labels, legend, etc.
        ax.set_xlabel('x (meters)')
        ax.set_ylabel('y (meters)')
        ax.set_title('Synchrotron Ring with Elements')
        ax.legend()
        ax.axis('equal')
        ax.grid(True)
        
        if plot_xlim is not None:
            ax.set_xlim(plot_xlim)
            plt.xlim(plot_xlim)
        
        if plot_ylim is not None:
            ax.set_ylim(plot_ylim)
            plt.ylim(plot_ylim)
        
        print("ax.get_xlim: ", ax.get_xlim())
        print("ax.get_ylim: ", ax.get_ylim())
        
        plt.tight_layout()

    def plot_particle_trajectory(self, start_idx=0, end_idx=None, plot_xlim=None, plot_ylim=None, save_label='0'):
        """Plot the particle trajectories (bird's-eye view) between specified indices."""
        total_turns = self.n_turns
        end_idx = end_idx if end_idx is not None else total_turns
        n_turns_to_plot = end_idx - start_idx

        # Prepare the plot
        fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
        
        # Plot the ring and elements into the same Axes
        self.plot_ring(ax=ax, plot_xlim=plot_xlim, plot_ylim=plot_ylim)

        # Now plot the particle trajectories
        colors = get_colors(self.num_particles)

        for idx in range(self.num_particles):
            total_points_per_turn = len(self.particles_states_x[idx]) // self.n_turns
            start_point = total_points_per_turn * start_idx
            end_point = total_points_per_turn * end_idx
            ax.plot(self.x_global_all[idx][start_point:end_point],
                    self.y_global_all[idx][start_point:end_point],
                    color=colors[idx])  # label=f'Particle {idx+1}'
            ax.plot(self.x_global_all[idx][start_point], self.y_global_all[idx][start_point],
                    marker='.', color=colors[idx], markersize=10, linestyle='None')

        ax.set_title(f'Particle Trajectories from Turn {start_idx} to {end_idx}')
        ax.legend()
        
        if plot_xlim is not None:
            ax.set_xlim(plot_xlim)
        
        if plot_ylim is not None:
            ax.set_ylim(plot_ylim)

        ax.set_aspect('equal')

        plt.tight_layout()
            
        plt.savefig(f"{self.figs_save_dir}/plot_particle_trajectory_{save_label}.eps", bbox_inches = 'tight', format='eps')

        plt.show()
        
        return fig, ax

    def plot_average_positions(self, cell_idx=0, window_size=5, start_idx=0, end_idx=None, save_label='0'):
        """
        Plot average positions vs revolution number with moving averages for a specific BPM.
        
        Parameters:
            cell_idx (int): Index of the BPM (FODO cell) to plot.
            window_size (int): Window size for the moving average.
            start_idx (int): Starting revolution index (inclusive).
            end_idx (int): Ending revolution index (exclusive). If None, plots until the last turn.
        """
        # Determine the total number of turns
        total_turns = self.n_turns
        end_idx = end_idx if end_idx is not None else total_turns
        n_turns_to_plot = end_idx - start_idx

        # Check window size
        if n_turns_to_plot < window_size:
            print(f"Window size {window_size} is larger than the number of turns {n_turns_to_plot}.")
            window_size = n_turns_to_plot
            if window_size == 0:
                print("No data to plot.")
                return

        # Extract positions for the specified BPM
        self.particles_avg_x_positions = []
        self.particles_avg_y_positions = []

        for idx in range(self.num_particles):
            avg_x_positions = self.bpm_readings['x'][idx, start_idx:end_idx, cell_idx]
            avg_y_positions = self.bpm_readings['y'][idx, start_idx:end_idx, cell_idx]
            self.particles_avg_x_positions.append(avg_x_positions)
            self.particles_avg_y_positions.append(avg_y_positions)

        # Compute overall average positions
        self.overall_avg_x_positions = np.mean(self.particles_avg_x_positions, axis=0)
        self.overall_avg_y_positions = np.mean(self.particles_avg_y_positions, axis=0)

        # Proceed with plotting
        plt.figure()
        colors = get_colors(self.num_particles)
        rev_numbers = np.arange(start_idx, end_idx)
        
        # Plot average x positions for each particle
        for idx in range(self.num_particles):
            avg_x_positions = self.particles_avg_x_positions[idx]
            if len(avg_x_positions) >= window_size:
                moving_avg_x = np.convolve(avg_x_positions, np.ones(window_size)/window_size, mode='valid')
                plt.plot(rev_numbers, avg_x_positions, color=colors[idx], label=f'Particle {idx+1}')
                plt.plot(rev_numbers[window_size - 1:], moving_avg_x, linestyle='--', color=colors[idx],
                        label=f'Moving Avg Particle {idx+1}')
            else:
                plt.plot(rev_numbers, avg_x_positions, color=colors[idx], label=f'Particle {idx+1}')

        plt.xlabel('Revolution Number')
        plt.ylabel('Horizontal Position x (State)')
        plt.title(f'Average Horizontal Positions at BPM {cell_idx} from Turn {start_idx} to {end_idx}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.figs_save_dir}/plot_average_positions_X_{save_label}.eps", bbox_inches = 'tight', format='eps')
        plt.show()

        # Plot average y positions for each particle
        plt.figure()
        for idx in range(self.num_particles):
            avg_y_positions = self.particles_avg_y_positions[idx]
            if len(avg_y_positions) >= window_size:
                moving_avg_y = np.convolve(avg_y_positions, np.ones(window_size)/window_size, mode='valid')
                plt.plot(rev_numbers, avg_y_positions, color=colors[idx], label=f'Particle {idx+1}')
                plt.plot(rev_numbers[window_size - 1:], moving_avg_y, linestyle='--', color=colors[idx],
                        label=f'Moving Avg Particle {idx+1}')
            else:
                plt.plot(rev_numbers, avg_y_positions, color=colors[idx], label=f'Particle {idx+1}')

        plt.xlabel('Revolution Number')
        plt.ylabel('Vertical Position y (State)')
        plt.title(f'Average Vertical Positions at BPM {cell_idx} from Turn {start_idx} to {end_idx}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.figs_save_dir}/plot_average_positions_Y_{save_label}.eps", bbox_inches = 'tight', format='eps')
        plt.show()

        # Plot overall average x positions
        plt.figure()
        if len(self.overall_avg_x_positions) >= window_size:
            overall_moving_avg_x = np.convolve(self.overall_avg_x_positions, np.ones(window_size)/window_size, mode='valid')
            plt.plot(rev_numbers, self.overall_avg_x_positions, label='Overall Avg x Position', color='k')
            plt.plot(rev_numbers[window_size - 1:], overall_moving_avg_x, linestyle='--', color='b',
                    label='Overall Moving Avg x')
        else:
            plt.plot(rev_numbers, self.overall_avg_x_positions, label='Overall Avg x Position', color='k')
        plt.xlabel('Revolution Number')
        plt.ylabel('Horizontal Position x (State)')
        plt.title(f'Overall Average Horizontal Position at BPM {cell_idx} from Turn {start_idx} to {end_idx}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.figs_save_dir}/plot_average_positions_X-overall_{save_label}.eps", bbox_inches = 'tight', format='eps')
        plt.show()

        # Plot overall average y positions
        plt.figure()
        if len(self.overall_avg_y_positions) >= window_size:
            overall_moving_avg_y = np.convolve(self.overall_avg_y_positions, np.ones(window_size)/window_size, mode='valid')
            plt.plot(rev_numbers, self.overall_avg_y_positions, label='Overall Avg y Position', color='k')
            plt.plot(rev_numbers[window_size - 1:], overall_moving_avg_y, linestyle='--', color='b',
                    label='Overall Moving Avg y')
        else:
            plt.plot(rev_numbers, self.overall_avg_y_positions, label='Overall Avg y Position', color='k')
        plt.xlabel('Revolution Number')
        plt.ylabel('Vertical Position y (State)')
        plt.title(f'Overall Average Vertical Position at BPM {cell_idx} from Turn {start_idx} to {end_idx}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.figs_save_dir}/plot_average_positions_Y-overall_{save_label}.eps", bbox_inches = 'tight', format='eps')
        plt.show()

    def plot_bpm_heatmaps(self, simulation_label='No Error', cell_idx=0, start_idx=0, end_idx=None, plot_xlim=None, plot_ylim=None, particle_idx=None, save_label='0', fontsize=14):
        """Plot combined heatmap of BPM measurements for a specific FODO cell."""
        if self.bpm_readings['x'] is None or self.bpm_readings['y'] is None:
            print("BPMs are not placed in the lattice.")
            return

        total_turns = self.n_turns
        end_idx = end_idx if end_idx is not None else total_turns

        if particle_idx:
            start_particle = particle_idx
            end_particle = particle_idx + 1
        else:
            start_particle = 0
            end_particle = self.bpm_readings['x'].shape[0]

        x_measurements = self.bpm_readings['x'][start_particle:end_particle, start_idx:end_idx, cell_idx].flatten()
        y_measurements = self.bpm_readings['y'][start_particle:end_particle, start_idx:end_idx, cell_idx].flatten()

        # Compute Center of Mass
        com_x = np.mean(x_measurements)
        com_y = np.mean(y_measurements)

        # Shift measurements to center around CoM
        # x_shifted = x_measurements - com_x
        # y_shifted = y_measurements - com_y

        x_shifted = x_measurements
        y_shifted = y_measurements

        # Create heatmap
        plt.figure(figsize=(8, 6))
        sns.histplot(x=x_shifted, y=y_shifted, bins=100, cmap='plasma', cbar=True, stat='density', alpha=1)
        plt.title(f'BPM Heatmap at Cell {cell_idx} Centered Around CoM ({simulation_label}) \n com_x={com_x} \n com_y={com_y}')
        plt.xlabel('Horizontal Position x relative to CoM (meters)', fontsize=fontsize)
        plt.ylabel('Vertical Position y relative to CoM (meters)', fontsize=fontsize)
        plt.tick_params(axis='both', labelsize=fontsize)
        plt.minorticks_on()
        # plt.scatter(0, 0, color='black', label='Center of Mass') #marker='X', s=100
        plt.gca().set_aspect('equal')
        plt.legend()
        
        if plot_xlim is not None:
            plt.gca().set_xlim(plot_xlim)
            plt.xlim(plot_xlim)
        
        if plot_ylim is not None:
            plt.gca().set_ylim(plot_ylim)
            plt.ylim(plot_ylim)
            
        plt.savefig(f"{self.figs_save_dir}/plot_bpm_heatmaps_{save_label}.eps", bbox_inches = 'tight', format='eps')
        plt.show()

    def plot_last_bpm_image(self, simulation_label='No Error', cell_idx=0, plot_xlim=None, plot_ylim=None, save_label='0'):
        """Plot heatmap of BPM measurements for the last turn at a specific FODO cell."""
        if self.bpm_readings['x'] is None or self.bpm_readings['y'] is None:
            print("BPMs are not placed in the lattice.")
            return

        last_turn = self.n_turns - 1
        x_last = self.bpm_readings['x'][:, last_turn, cell_idx]
        y_last = self.bpm_readings['y'][:, last_turn, cell_idx]

        # Compute Center of Mass
        com_x = np.mean(x_last)
        com_y = np.mean(y_last)

        # Shift measurements to center around CoM
        x_shifted = x_last - com_x
        y_shifted = y_last - com_y

        # Create heatmap
        plt.figure(figsize=(8, 6))
        sns.histplot(x=x_shifted, y=y_shifted, bins=50, cmap='plasma', cbar=True, stat='density', alpha=1)
        plt.title(f'BPM Heatmap at Cell {cell_idx} - Last Turn Centered Around CoM ({simulation_label})')
        plt.xlabel('Horizontal Position x relative to CoM (meters)')
        plt.ylabel('Vertical Position y relative to CoM (meters)')
        plt.scatter(0, 0, color='white', marker='X', s=100, label='Center of Mass')
        plt.gca().set_aspect('equal')
        plt.legend()
        
        if plot_xlim is not None:
            plt.gca().set_xlim(plot_xlim)
            plt.xlim(plot_xlim)
        
        if plot_ylim is not None:
            plt.gca().set_ylim(plot_ylim)
            plt.ylim(plot_ylim)
            
        plt.savefig(f"{self.figs_save_dir}/plot_last_bpm_image_{save_label}.eps", bbox_inches = 'tight', format='eps')

        plt.show()

    def plot_bpm_comparison_last_images(self, simulator_no_error, simulator_with_error, cell_idx=0, particles='all', save_label='0', msg='With vs Without Quadrupole Error'):
        """Compare the last BPM heatmaps between two simulations for a specific FODO cell.
        particles: 'all' | 'all_mean'
        """
        if simulator_no_error.bpm_readings['x'] is None or simulator_no_error.bpm_readings['y'] is None:
            print("BPMs are not placed in the lattice for the first simulator.")
            return
        if simulator_with_error.bpm_readings['x'] is None or simulator_with_error.bpm_readings['y'] is None:
            print("BPMs are not placed in the lattice for the second simulator.")
            return

        # Extract the last turn's BPM readings for the specified cell
        last_turn_no_error = simulator_no_error.n_turns - 1
        last_turn_with_error = simulator_with_error.n_turns - 1

        if particles == 'all':
            x_no_error_all = simulator_no_error.bpm_readings['x'][:, 0:last_turn_no_error, cell_idx]
            y_no_error_all = simulator_no_error.bpm_readings['y'][:, 0:last_turn_no_error, cell_idx]
            x_with_error_all = simulator_with_error.bpm_readings['x'][:, 0:last_turn_with_error, cell_idx]
            y_with_error_all = simulator_with_error.bpm_readings['y'][:, 0:last_turn_with_error, cell_idx]

            x_no_error_all = x_no_error_all.flatten()
            y_no_error_all = y_no_error_all.flatten()
            x_with_error_all = x_with_error_all.flatten()
            y_with_error_all = y_with_error_all.flatten()

            x_no_error = x_no_error_all
            y_no_error = y_no_error_all
            x_with_error = x_with_error_all
            y_with_error =  y_with_error_all       

        elif particles == 'all_mean':
            x_no_error_mean = simulator_no_error.bpm_readings['x'][:, 0:last_turn_no_error, cell_idx].mean(axis=0)
            y_no_error_mean = simulator_no_error.bpm_readings['y'][:, 0:last_turn_no_error, cell_idx].mean(axis=0)
            x_with_error_mean = simulator_with_error.bpm_readings['x'][:, 0:last_turn_with_error, cell_idx].mean(axis=0)
            y_with_error_mean = simulator_with_error.bpm_readings['y'][:, 0:last_turn_with_error, cell_idx].mean(axis=0)

            x_no_error_mean = x_no_error_mean.flatten()
            y_no_error_mean = y_no_error_mean.flatten()
            x_with_error_mean = x_with_error_mean.flatten()
            y_with_error_mean = y_with_error_mean.flatten()
            
            x_no_error = x_no_error_mean
            y_no_error = y_no_error_mean
            x_with_error = x_with_error_mean
            y_with_error =  y_with_error_mean

        # x_no_error_ravg = self.running_average_numpy(x_no_error, window_size=len(x_no_error) - int(len(x_no_error) * 0.1))
        # y_no_error_ravg = self.running_average_numpy(y_no_error, window_size=len(y_no_error) - int(len(x_no_error) * 0.1))
        # x_with_error_ravg = self.running_average_numpy(x_with_error, window_size=len(x_with_error) - int(len(x_no_error) * 0.1))
        # y_with_error_ravg = self.running_average_numpy(y_with_error, window_size=len(y_with_error) - int(len(x_no_error) * 0.1))

        # Compute Center of Mass for both simulations
        com_x_no_error = np.mean(x_no_error)
        com_y_no_error = np.mean(y_no_error)
        com_x_with_error = np.mean(x_with_error)
        com_y_with_error = np.mean(y_with_error)

        # Compute differences
        delta_x = com_x_with_error - com_x_no_error
        delta_y = com_y_with_error - com_y_no_error

        print("plot_bpm_comparison_last_images()/")
        print(f"\t CoM No error: X = {com_x_no_error}, Y = {com_y_no_error}")
        print(f"\t CoM With Error: X = {com_x_with_error}, Y = {com_y_with_error}")
        print('----')
        print(f"\t f'ΔX = {delta_x:.7f} m, {delta_x * 1e6:.2f} micron")
        print(f"\t f'ΔY = {delta_y:.7f} m, {delta_y * 1e6:.2f} micron")
        print('----')

        # **Retrieve Epsilon Measurements**
        # Ensure that both simulators have calculated beam sizes
        if simulator_no_error.epsilon_horizontal is None or simulator_no_error.epsilon_vertical is None:
            simulator_no_error.calculate_beam_size()
        if simulator_with_error.epsilon_horizontal is None or simulator_with_error.epsilon_vertical is None:
            simulator_with_error.calculate_beam_size()

        # Prepare the plot with a darker background
        plt.figure(figsize=(12, 8), facecolor='black')
        ax = plt.gca()
        ax.set_facecolor('black')  # Set axes background

        # Determine common bin ranges based on both datasets
        all_x = np.concatenate([x_no_error, x_with_error])
        all_y = np.concatenate([y_no_error, y_with_error])
        x_min, x_max = np.min(all_x), np.max(all_x)
        y_min, y_max = np.min(all_y), np.max(all_y)

        # Determine bin size based on the larger range to ensure square bins
        range_x = x_max - x_min
        range_y = y_max - y_min
        max_range = max(range_x, range_y)

        # Define desired bin size (meters). Adjust this value to change bin size.
        desired_bin_size = max_range / 50  # Example: 50 bins along the largest axis

        # Create bin edges with the same bin size for both axes
        x_bins = np.arange(x_min, x_max + desired_bin_size, desired_bin_size)
        y_bins = np.arange(y_min, y_max + desired_bin_size, desired_bin_size)

        # Plot heatmap for without_error simulation

        sns.histplot(
            x=x_no_error,
            y=y_no_error,
            bins=[x_bins, y_bins],
            cmap='Blues',
            cbar=True,
            stat='density',
            alpha=0.5,
            label='No Error',
            edgecolor=None
        )

        # Plot heatmap for with_error simulation
        sns.histplot(
            x=x_with_error,
            y=y_with_error,
            bins=[x_bins,y_bins],
            cmap='Reds',
            cbar=True,
            stat='density',
            alpha=0.5,
            label='With Quadrupole Error',
            edgecolor=None
        )

        # Overlay the CoM points
        plt.scatter(com_x_no_error, com_y_no_error, color='cyan', marker='o', s=100, label='CoM No Error')
        plt.scatter(com_x_with_error, com_y_with_error, color='magenta', marker='X', s=100, label='CoM With Error')

        # Plot horizontal dashed line representing ΔX
        plt.plot(
            [com_x_no_error, com_x_with_error], 
            [com_y_no_error, com_y_no_error], 
            color='yellow', 
            linestyle='--', 
            linewidth=2, 
            label=f'ΔX = {delta_x * 1e6:.2f} micron'
        )

        # Plot vertical dashed line representing ΔY
        plt.plot(
            [com_x_with_error, com_x_with_error], 
            [com_y_no_error, com_y_with_error], 
            color='lime', 
            linestyle='--', 
            linewidth=2, 
            label=f'ΔY = {delta_y * 1e6:.2f} micron'
        )

        # **Add Epsilon Measurements as Annotations**
        plt.text(
            x_min + 0.05 * range_x, y_max - 0.1 * range_y, 
            f"Epsilon Horizontal (No Error): {simulator_no_error.epsilon_horizontal:.6f} micron", 
            color='cyan', fontsize=12
        )
        plt.text(
            x_min + 0.05 * range_x, y_max - 0.15 * range_y,
            f"Epsilon Vertical (No Error): {simulator_no_error.epsilon_vertical:.6f} micron", 
            color='cyan', fontsize=12
        )
        plt.text(
            x_min + 0.05 * range_x, y_max - 0.2 * range_y, 
            f"Epsilon Horizontal (With Error): {simulator_with_error.epsilon_horizontal:.6f} micron", 
            color='magenta', fontsize=12
        )
        plt.text(
            x_min + 0.05 * range_x, y_max - 0.25 * range_y, 
            f"Epsilon Vertical (With Error): {simulator_with_error.epsilon_vertical:.6f} micron", 
            color='magenta', fontsize=12
        )

        # Customize plot aesthetics
        plt.title(f'BPM Heatmap Comparison at Cell {cell_idx}: {msg}', color='white', fontsize=16)
        plt.xlabel('Horizontal Position x (micron)', color='white', fontsize=14)
        plt.ylabel('Vertical Position y (micron)', color='white', fontsize=14)

        # Set axis labels color to white for visibility on dark background
        plt.xticks(color='white')
        plt.yticks(color='white')

        # Add legend with white text
        legend = plt.legend(facecolor='gray', edgecolor='white', framealpha=0.7)
        for text in legend.get_texts():
            text.set_color("white")

        # Set limits for better visualization
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        # Set aspect ratio to equal to ensure square bins
        plt.axis('equal')

        plt.grid(True, color='gray', alpha=0.5)
        plt.tight_layout()

        plt.savefig(f"{self.figs_save_dir}/plot_bpm_comparison_last_images_{particles}_{save_label}.eps", bbox_inches = 'tight', format='eps')
        plt.show()

    def plot_all_bpm_heatmap(self, simulation_label='No Error', start_idx=0, end_idx=None, plot_xlim=None, plot_ylim=None, bins=100, save_label='0'):
        """Plot combined heatmap of BPM measurements from all BPMs."""
        if self.bpm_readings['x'] is None or self.bpm_readings['y'] is None:
            print("BPMs are not placed in the lattice.")
            return

        total_turns = self.n_turns
        end_idx = end_idx if end_idx is not None else total_turns

        # Collect all x and y measurements from all BPMs
        x_measurements = self.bpm_readings['x'][:, start_idx:end_idx, :].flatten()
        y_measurements = self.bpm_readings['y'][:, start_idx:end_idx, :].flatten()

        # Compute Center of Mass
        com_x = np.mean(x_measurements)
        com_y = np.mean(y_measurements)

        # # Shift measurements to center around CoM
        # x_shifted = x_measurements - com_x
        # y_shifted = y_measurements - com_y
        
        x_shifted = x_measurements
        y_shifted = y_measurements

        # Create heatmap
        plt.figure(figsize=(8, 6))
        sns.histplot(x=x_shifted, y=y_shifted, bins=100, cmap='plasma', cbar=True, stat='density', alpha=1)
        plt.title(f'Combined BPM Heatmap from All BPMs ({simulation_label})\ncom_x={com_x}, com_y={com_y}')
        plt.xlabel('Horizontal Position x relative to CoM (meters)')
        plt.ylabel('Vertical Position y relative to CoM (meters)')
        plt.gca().set_aspect('equal')
        plt.legend()
        
        if plot_xlim is not None:
            plt.gca().set_xlim(plot_xlim)
            plt.xlim(plot_xlim)
        
        if plot_ylim is not None:
            plt.gca().set_ylim(plot_ylim)
            plt.ylim(plot_ylim)

        plt.savefig(f"{self.figs_save_dir}/plot_all_bpm_heatmap_{save_label}.eps", bbox_inches = 'tight', format='eps')

        plt.show()

    def plot_phase_space_diagram(
        self,
        first_axis='x',
        second_axis='y',
        cell_idx=0,
        start_idx=0,
        end_idx=None,
        particle_idx=None,
        save_label='0',
        fontsize=14,
        plot_all=False,
        bins=150
    ):
        """
        Plot the phase space heatmap (first_axis vs. second_axis) at one BPM (cell_idx),
        or if plot_all=True, then for all BPMs in subplots, over a given turn range.

        Parameters:
            first_axis (str):  One of ['x', 'y', 'xp', 'yp']. Default='x'.
            second_axis (str): One of ['x', 'y', 'xp', 'yp']. Default='y'.
            cell_idx (int):    Index of the BPM (FODO cell) to plot if plot_all=False.
            start_idx (int):   Starting revolution index (inclusive). Default=0.
            end_idx (int):     Ending revolution index (exclusive). If None, uses last turn.
            particle_idx (int or None): Which particle to plot. If None, flattens all. 
            save_label (str):  Label appended to the saved figure filename. Default='0'.
            fontsize (int):    Font size for plot labels and titles. Default=14.
            plot_all (bool):   If True, plot all BPMs in one figure (subplots). 
                            If False, plot only the single cell_idx. Default=False.
        """
        # Check if BPM readings are available
        if (self.bpm_readings['x'] is None) or (self.bpm_readings['xp'] is None):
            print("BPM readings are not available.")
            return

        total_turns = self.n_turns
        end_idx = end_idx if (end_idx is not None) else total_turns

        # Number of BPMs
        n_BPMs = self.bpm_readings['x'].shape[2]

        # --------------------------------------------------
        #  Case 1) plot_all = False => Single BPM as before
        # --------------------------------------------------
        if not plot_all:
            # Extract data for the specified axes, BPM, turn range
            if particle_idx is None:
                first_axis_data = self.bpm_readings[first_axis][:, start_idx:end_idx, cell_idx].flatten()
                second_axis_data = self.bpm_readings[second_axis][:, start_idx:end_idx, cell_idx].flatten()
            else:
                first_axis_data = self.bpm_readings[first_axis][particle_idx, start_idx:end_idx, cell_idx]
                second_axis_data = self.bpm_readings[second_axis][particle_idx, start_idx:end_idx, cell_idx]

            # Create the heatmap
            plt.figure(figsize=(8, 6))
            
            
            hist, xedges, yedges, im = plt.hist2d(
                first_axis_data, 
                second_axis_data, 
                bins=bins, 
                cmap='jet', 
                density=True)
            plt.grid(True, linestyle="--", alpha=0.3)
        
            # Colorbar for intensity reference
            cbar = plt.colorbar(im)
            cbar.set_label("Density of Particles", fontsize=12)

            
            # sns.histplot(
            #     x=first_axis_data,
            #     y=second_axis_data,
            #     bins=100,
            #     cmap=cm.jet,
            #     stat="density"
            # )

            
            plt.xlabel(f'{first_axis} (meters)', fontsize=fontsize)
            plt.ylabel(f'{second_axis} (radians)', fontsize=fontsize)
            plt.title(
                f'Phase Space Heatmap ({first_axis} vs. {second_axis}) at BPM {cell_idx}',
                fontsize=fontsize
            )
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)

            # Save the figure
            plt.savefig(
                f"{self.figs_save_dir}/plot_phase_space_heatmap_{first_axis}-Vs-{second_axis}_{save_label}.eps",
                bbox_inches='tight',
                format='eps'
            )
            plt.show()

        # --------------------------------------------------
        #  Case 2) plot_all = True => Subplots for each BPM
        # --------------------------------------------------
        else:
            # Determine subplot layout
            n_cols = 3
            n_rows = int(np.ceil(n_BPMs / n_cols))

            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(5 * n_cols, 4 * n_rows),
                sharex=False, sharey=False
            )
            axes = axes.flatten()  # Flatten for easy indexing

            # Loop over all BPM cells
            for bpm_idx in range(n_BPMs):
                ax = axes[bpm_idx]

                # Collect data for each BPM in the same manner
                if particle_idx is None:
                    first_axis_data = self.bpm_readings[first_axis][:, start_idx:end_idx, bpm_idx].flatten()
                    second_axis_data = self.bpm_readings[second_axis][:, start_idx:end_idx, bpm_idx].flatten()
                else:
                    first_axis_data = self.bpm_readings[first_axis][particle_idx, start_idx:end_idx, bpm_idx]
                    second_axis_data = self.bpm_readings[second_axis][particle_idx, start_idx:end_idx, bpm_idx]

                # # Plot histogram in each subplot
                # sns.histplot(
                #     x=first_axis_data,
                #     y=second_axis_data,
                #     bins=100,
                #     cmap="jet",
                #     stat="density",
                #     ax=ax
                # )
                hist, xedges, yedges, im = ax.hist2d(
                    first_axis_data, 
                    second_axis_data, 
                    bins=bins, 
                    cmap='jet', 
                    density=True)
                ax.grid(True, linestyle="--", alpha=0.3)
                            
                ax.set_xlabel(f'{first_axis} (m)', fontsize=fontsize-2)
                ax.set_ylabel(f'{second_axis}', fontsize=fontsize-2)
                ax.set_title(f'BPM {bpm_idx}', fontsize=fontsize-2)
                ax.tick_params(axis='both', labelsize=fontsize-2)
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)

            # Colorbar for intensity reference
            cbar_ax = fig.add_axes([1, 0.1, .03, .7])
            cbar = plt.colorbar(im, cax=cbar_ax)
            cbar.set_label("Density of Particles", fontsize=12)

            # Hide any unused subplots if n_BPMs < n_rows*n_cols
            for ax_idx in range(n_BPMs, n_rows * n_cols):
                fig.delaxes(axes[ax_idx])

            # Give an overall title and save
            fig.suptitle(
                f'Phase Space Heatmaps ({first_axis} vs. {second_axis}) at All BPMs\nTurns {start_idx}–{end_idx}',
                fontsize=fontsize
            )
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(
                f"{self.figs_save_dir}/plot_phase_space_heatmap_ALL_{first_axis}-Vs-{second_axis}_{save_label}.eps",
                bbox_inches='tight',
                format='eps'
            )
            plt.show()

    def plot_particle_bpm_readings(self, particle_idx=0, start_idx=0, end_idx=None, save_label='0'):
        """
        Plot BPM readings (x and y) for a specific particle over all BPMs and revolutions,
        with x-axis ticks based on betatron tune Qx, Qy.

        After n full betatron oscillations in X, we have n = Qx * (# of ring turns).
        So 1 "X revolution" happens every (1/Qx) ring turns in the real machine.

        Parameters:
            particle_idx (int): Which particle to plot.
            start_idx (int):    Starting revolution index (inclusive).
            end_idx (int):      Ending revolution index (exclusive). If None, plots until the last turn.
            save_label (str):   Label for the saved figure name.
        """
        if self.bpm_readings['x'] is None or self.bpm_readings['y'] is None:
            print("BPM readings are not available.")
            return

        total_turns = self.n_turns
        end_idx = end_idx if end_idx is not None else total_turns

        # Extract the data for the chosen particle and turn range
        x_data = self.bpm_readings['x'][particle_idx, start_idx:end_idx, :]
        y_data = self.bpm_readings['y'][particle_idx, start_idx:end_idx, :]

        # x_data, y_data each has shape (revs, bpms), 
        # where revs = number of turns in [start_idx, end_idx]
        # and bpms = self.n_FODO (if there's one BPM per cell).
        revs, bpms = x_data.shape

        # Flatten so that the x-axis is one long series of measurements
        # The total length = revs * bpms
        x_data = x_data.flatten()
        y_data = y_data.flatten()

        # ------------------------------------------------------------
        # Plot X data, with x-axis ticks based on the horizontal tune Qx
        # ------------------------------------------------------------
        plt.figure(figsize=(20, 0.6))
        plt.plot(x_data, '-')
        plt.ylabel('X [m]')
        plt.xlabel('Betatron Oscillations in X (integer ticks)')

        # Generate ticks for each integer number of X betatron cycles:
        # 1 X cycle = 1/Qx ring turns, so after n cycles => n/Qx turns.
        # In the flattened array: index_in_flat = (n/Qx) * bpms
        n_max_x_cycles = int(np.floor(self.Qx * revs))  # how many full X cycles within 'revs' ring turns
        x_tick_positions = []
        x_tick_labels = []
        for n in range(n_max_x_cycles + 1):
            # ring_turn is the actual ring turn number. We place a tick at that.
            ring_turn = n / self.Qx  
            if ring_turn <= revs:
                idx_in_flat = ring_turn * bpms
                x_tick_positions.append(idx_in_flat)
                x_tick_labels.append(str(n))  # label with integer cycle count

        plt.xticks(x_tick_positions, x_tick_labels)
        plt.title(f'X - BPM data flattened\nTurns={revs}, Qx={self.Qx:.5f}')
        plt.grid(True)

        plt.savefig(f"{self.figs_save_dir}/plot_particle_bpm_readings_X_{save_label}.eps",
                    bbox_inches='tight', format='eps')
        plt.show()

        # ------------------------------------------------------------
        # Plot Y data, with x-axis ticks based on the vertical tune Qy
        # ------------------------------------------------------------
        plt.figure(figsize=(20, 0.6))
        plt.plot(y_data, '-')
        plt.ylabel('Y [m]')
        plt.xlabel('Betatron Oscillations in Y (integer ticks)')

        n_max_y_cycles = int(np.floor(self.Qy * revs))
        y_tick_positions = []
        y_tick_labels = []
        for n in range(n_max_y_cycles + 1):
            ring_turn = n / self.Qy
            if ring_turn <= revs:
                idx_in_flat = ring_turn * bpms
                y_tick_positions.append(idx_in_flat)
                y_tick_labels.append(str(n))

        plt.xticks(y_tick_positions, y_tick_labels)
        plt.title(f'Y - BPM data flattened\nTurns={revs}, Qy={self.Qy:.5f}')
        plt.grid(True)

        plt.savefig(f"{self.figs_save_dir}/plot_particle_bpm_readings_Y_{save_label}.eps",
                    bbox_inches='tight', format='eps')
        plt.show()


    def compare_bpm_signal_vs_bpm_number(
        self,
        other_simulator,
        turn='last',
        plane='horizontal',
        average_over_particles=True,
        plot_difference=True,
        figsize=(8, 5),
        save_label='0',
        fontsize=9
    ):
        """
        Compare BPM signals (x or y) vs. BPM index for two simulations:
        - self: (e.g., the 'no error' simulator)
        - other_simulator: (e.g., the 'with error' simulator)

        Shows how a misalignment or tilt changes the beam orbit across all BPMs
        at a given turn.

        Parameters
        ----------
        other_simulator : SynchrotronSimulator
            Another SynchrotronSimulator instance to compare against. 
            Typically, you pass a simulator that includes quadrupole/dipole errors.
        turn : int or 'last'
            Which turn index to compare. If 'last', uses the final turn.
        plane : str
            Which plane to compare: 'horizontal' (x) or 'vertical' (y).
        average_over_particles : bool
            If True, plot the mean orbit across all particles at each BPM.
            If False, plot each particle's data (can be crowded for large ensembles).
        plot_difference : bool
            If True, also plot a small inset showing (with_error - no_error).
        figsize : tuple
            Figure size in inches, e.g., (8, 5).

        Returns
        -------
        fig, ax : The Matplotlib Figure and Axes for further customization.
        """

        # ---- 1) Retrieve BPM data from both simulators -----------
        if plane.lower() == 'horizontal':
            data_self  = self.bpm_readings['x']   # shape (num_particles, n_turns, n_BPMs)
            data_other = other_simulator.bpm_readings['x']
            y_label = "Horizontal Displacement x (m)"
        elif plane.lower() == 'vertical':
            data_self  = self.bpm_readings['y']
            data_other = other_simulator.bpm_readings['y']
            y_label = "Vertical Displacement y (m)"
        else:
            raise ValueError("plane must be 'horizontal' or 'vertical'")

        # ---- 2) Determine turn index to plot ---------------------
        if turn == 'last':
            turn_idx = min(self.n_turns, other_simulator.n_turns) - 1
        else:
            turn_idx = int(turn)

        # Safety check
        if turn_idx < 0 or turn_idx >= self.n_turns or turn_idx >= other_simulator.n_turns:
            raise ValueError(f"Requested turn {turn_idx} is out of range for at least one simulator.")

        # ---- 3) Extract data for the chosen turn ------------------
        # shape => (num_particles, n_BPMs)
        turn_data_self  = data_self[:,  turn_idx, :]
        turn_data_other = data_other[:, turn_idx, :]

        # ---- 4) Create figure & axes -----------------------------
        fig, ax = plt.subplots(figsize=figsize)

        # BPM indices => 0..(n_FODO - 1)
        # We assume both sims have the same number of BPMs (same n_FODO)
        if self.n_FODO != other_simulator.n_FODO:
            print("[Warning] The two simulators have different n_FODO values. "
                "Their BPM signals may not match up one-to-one.")
        bpm_indices = np.arange(self.n_FODO)

        # ---- 5) Plot signals --------------------------------------
        if average_over_particles:
            # Mean orbit across all particles
            mean_self  = np.mean(turn_data_self,  axis=0)  # shape => (n_BPMs,)
            mean_other = np.mean(turn_data_other, axis=0)

            ax.plot(bpm_indices, mean_self,  marker='o', linestyle='-',  color='blue', label='No Error (self)')
            ax.plot(bpm_indices, mean_other, marker='s', linestyle='--', color='red',  label='With Error (other)')

            ax.set_title(
                f"{plane.capitalize()} BPM Signals vs BPM Index\n"
                f"(Turn={turn_idx}, Avg over {self.num_particles} particles)",
                fontsize=fontsize
            )
        else:
            # Plot each particle curve
            n_p_self  = turn_data_self.shape[0]
            n_p_other = turn_data_other.shape[0]

            for pid in range(n_p_self):
                ax.plot(bpm_indices, turn_data_self[pid, :],
                        marker='o', linestyle='-', alpha=0.2, color='blue')
            for pid in range(n_p_other):
                ax.plot(bpm_indices, turn_data_other[pid, :],
                        marker='s', linestyle='--', alpha=0.2, color='red')

            ax.set_title(
                f"{plane.capitalize()} BPM Signals vs BPM Index\n"
                f"(Turn={turn_idx}, All Particles)",
                fontsize=fontsize
            )

        ax.set_xlabel("BPM Index (0 … n_FODO-1)", fontsize=fontsize)
        ax.set_ylabel(y_label, fontsize=fontsize)
        ax.grid(True)
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.legend()

        # ---- 6) Optional: Plot difference (with_error - no_error) --
        if plot_difference and average_over_particles:
            difference = mean_other - mean_self
            ax_inset = ax.inset_axes([0.58, 0.08, 0.38, 0.32])  # [x0, y0, width, height]
            ax_inset.plot(bpm_indices, difference, marker='d', color='green')
            ax_inset.set_title("Difference (Error - No Error)", fontsize=int(fontsize/2))
            ax_inset.set_xlabel("BPM Index", fontsize=int(fontsize/1.5))
            ax_inset.set_ylabel("Δ Orbit (m)", fontsize=int(fontsize/1.5))
            ax_inset.tick_params(axis='both', labelsize=int(fontsize/1.5))
            ax_inset.grid(True)

        plt.tight_layout()

        plt.savefig(f"{self.figs_save_dir}/compare_bpm_signal_vs_bpm_number_{save_label}.eps", bbox_inches = 'tight', format='eps')
        
        plt.show()
        return fig, ax


    def plot_initial_positions_heatmap(self, bins=200):
        """
        Plots a heatmap of the initial transverse positions (X, Y) of all particles
        in the storage ring before any motion occurs.

        Parameters:
        - bins (int): Number of bins for the 2D histogram.
        """
        # Extract initial positions
        x_init = self.bpm_readings['x'][:,  0, 0]
        y_init = self.bpm_readings['y'][:,  0, 0]
        
        # Create the 2D histogram
        plt.figure(figsize=(8, 8))
        hist, xedges, yedges, im = plt.hist2d(x_init, y_init, bins=bins, cmap='jet', density=True)

        # Colorbar for intensity reference
        cbar = plt.colorbar(im)
        cbar.set_label("Density of Particles", fontsize=12)

        # Labels and Title
        plt.xlabel("X Position [m]")
        plt.ylabel("Y Position [m]")
        plt.title("Initial Particle Positions Heatmap")
        plt.grid(True, linestyle="--", alpha=0.3)

        plt.tight_layout()
        
        # Save as EPS format
        plt.savefig("initial_positions_heatmap.eps", format='eps', bbox_inches='tight')

        # Show the plot
        plt.show()

    def plot_betatron_oscillations_TURN_FLATTEN(self, start_turn=0, end_turn=None, 
                                    n_particles=50, figsize=(20, 12), save_label=''):
        """
        Plot betatron oscillations with cyclic time-series visualization across BPMs and turns

        """
        # Handle turn range
        if end_turn is None: 
            end_turn = self.n_turns - 1
        start_turn = max(0, start_turn)
        end_turn = min(end_turn, self.n_turns - 1)
        turns = np.arange(start_turn, end_turn + 1)
        n_bpms = self.bpm_readings['x'].shape[2]

        # Get data and flatten across turns
        def get_flattened_data(simulator, data_key):
            data = getattr(simulator.bpm_readings[data_key], 'copy')()
            selected = data[:, start_turn:end_turn+1, :]  # (particles, turns, bpms)
            return selected.reshape(selected.shape[0], -1)  # Flatten turns and bpms

        x_self = get_flattened_data(self, 'x')
        y_self = get_flattened_data(self, 'y')

        # Create cyclic time axis
        time_points = np.arange(x_self.shape[1])
        n_total_points = len(time_points)
        
        # Calculate full oscillation periods
        turns_per_osc_x = int(round(1/self.Qx))
        turns_per_osc_y = int(round(1/self.Qy))
        points_per_osc_x = turns_per_osc_x * n_bpms
        points_per_osc_y = turns_per_osc_y * n_bpms

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Common styling
        label_font = 18
        tick_font = 15
        alpha = 0.15
        lw = 0.5
        
        # Plot horizontal plane
        for i in range(min(n_particles, x_self.shape[0])):
            ax1.plot(time_points, x_self[i], 'b', alpha=alpha, lw=lw)
        
        # Plot average horizontal oscillations
        x_avg = np.mean(x_self, axis=0)  # Average across particles
        ax1.plot(time_points, x_avg, 'b', alpha=0.5, lw=6, label='Average')
        
        # # Add oscillation markers
        for osc in range(points_per_osc_x, n_total_points, points_per_osc_x):
            ax1.axvline(osc, color='navy', ls='--', alpha=0.5, lw=3)
            ax1.text(osc, ax1.get_ylim()[1] * 0.95, f'{osc // n_bpms} Turns', 
                    color='navy', fontsize=label_font-4, ha='center', va='top', rotation=90)
        
        # Plot vertical plane
        for i in range(min(n_particles, y_self.shape[0])):
            ax2.plot(time_points, y_self[i], 'g', alpha=alpha, lw=lw)
        
        # Plot average vertical oscillations
        y_avg = np.mean(y_self, axis=0)  # Average across particles
        ax2.plot(time_points, y_avg, 'g', alpha=0.5, lw=6, label='Average')
        
        # Add oscillation markers
        for osc in range(points_per_osc_y, n_total_points, points_per_osc_y):
            ax2.axvline(osc, color='darkgreen', ls='--', alpha=0.5, lw=3)
            ax2.text(osc, ax2.get_ylim()[1] * 0.95, f'{osc // n_bpms} Turns', 
                    color='darkgreen', fontsize=label_font-4, ha='center', va='top', rotation=90)


        # Formatting
        ax1.set_ylabel('Horizontal Position [m]', fontsize=label_font)
        ax2.set_ylabel('Vertical Position [m]', fontsize=label_font)
        ax2.set_xlabel('Cyclic Time (BPM Index × Turn Number)', fontsize=label_font)
        
        # Add tune annotations
        ax1.text(0.5, 0.95, f'Qx = {self.Qx:.3f}\n1/Qx = {turns_per_osc_x} turns',
                transform=ax1.transAxes, ha='right', va='top',
                fontsize=label_font-2, bbox=dict(facecolor='white', alpha=0.8))
        
        ax2.text(0.5, 0.95, f'Qy = {self.Qy:.3f}\n1/Qy = {turns_per_osc_y} turns',
                transform=ax2.transAxes, ha='right', va='top',
                fontsize=label_font-2, bbox=dict(facecolor='white', alpha=0.8))

        # Add turn separators with annotations
        for t in range(n_bpms, n_total_points, n_bpms):
            ax1.axvline(t, color='gray', ls=':', alpha=0.3, lw=5)
            ax2.axvline(t, color='gray', ls=':', alpha=0.3, lw=5)
            if t % (n_bpms * 5) == 0:  # Annotate every 5 turns to avoid clutter
                ax1.text(t, ax1.get_ylim()[0] * 0.95, f'Turn {t // n_bpms}', 
                        color='gray', fontsize=label_font-4, ha='center', va='bottom', rotation=90)
                ax2.text(t, ax2.get_ylim()[0] * 0.95, f'Turn {t // n_bpms}', 
                        color='gray', fontsize=label_font-4, ha='center', va='bottom', rotation=90)


        # Final touches
        for ax in [ax1, ax2]:
            ax.grid(True)
            ax.tick_params(axis='both', labelsize=tick_font)
            ax.legend(loc='upper right', fontsize=label_font-2)
        
        fig.suptitle(f'Cyclic Betatron Oscillations (Turns {start_turn}-{end_turn}, {n_bpms} BPMs)', 
                    fontsize=label_font+2, y=1.02)
        plt.tight_layout()
        plt.savefig(f"{self.figs_save_dir}/cyclic_betatron_{save_label}.png", 
                format='png', bbox_inches='tight')
        plt.show()

    def plot_betatron_oscillations_BPM(self, other_simulator, start_turn=0, end_turn=None, 
                                n_particles=100, figsize=(20, 12), save_label=''):
        """
        Plot closed orbits and betatron oscillations for X and Y separately across multiple turns.
        
        Parameters:
            other_simulator (SynchrotronSimulator): Simulator WITH errors
            start_turn (int): First turn to include in analysis
            end_turn (int): Last turn to include in analysis
            n_particles (int): Number of particle trajectories to plot
            figsize (tuple): Figure size
            save_label (str): Suffix for figure filename
        """
        # Handle turn range
        if end_turn is None:
            end_turn = self.n_turns - 1
        start_turn = max(0, start_turn)
        end_turn = min(self.n_turns - 1, end_turn)
        
        if start_turn > end_turn:
            start_turn, end_turn = end_turn, start_turn  # Swap if reversed

        # Get data from both simulators
        def get_data(simulator, data_key):
            data = getattr(simulator.bpm_readings[data_key], 'copy')()
            return data[:, start_turn:end_turn+1, :]  # Shape: (particles, turns, bpms)

        # Get X and Y data for both simulators
        x_self = get_data(self, 'x')
        y_self = get_data(self, 'y')
        x_other = get_data(other_simulator, 'x')
        y_other = get_data(other_simulator, 'y')

        # Average over selected turns
        x_self_avg = np.mean(x_self, axis=1)
        y_self_avg = np.mean(y_self, axis=1)
        x_other_avg = np.mean(x_other, axis=1)
        y_other_avg = np.mean(y_other, axis=1)

        # Calculate closed orbits (mean across particles and turns)
        x_co_self = np.mean(x_self_avg, axis=0)
        y_co_self = np.mean(y_self_avg, axis=0)
        x_co_other = np.mean(x_other_avg, axis=0)
        y_co_other = np.mean(y_other_avg, axis=0)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        bpm_indices = np.arange(len(x_co_self))

        # Common styling parameters
        label_font = 18
        tick_font = 15
        alpha = 0.2
        lw = 0.7

        # Plot X plane
        # Non-distorted
        ax1.plot(bpm_indices, x_co_self*1e6, '-', color='blue', 
                label='Non-Distorted Closed Orbit', linewidth=3)
        for i in range(min(n_particles, x_self_avg.shape[0])):
            ax1.plot(bpm_indices, x_self_avg[i,:]*1e6, color='blue', alpha=alpha, linewidth=lw)

        # Distorted
        ax1.plot(bpm_indices, x_co_other*1e6, '--', color='red', 
                label='Distorted Closed Orbit', linewidth=3)
        for i in range(min(n_particles, x_other_avg.shape[0])):
            ax1.plot(bpm_indices, x_other_avg[i,:]*1e6, color='red', alpha=alpha, linewidth=lw)

        ax1.set_ylabel('Horizontal Position [μm]', fontsize=label_font)
        ax1.grid(True)
        ax1.legend(fontsize=label_font)
        ax1.tick_params(axis='both', labelsize=tick_font)

        # Plot Y plane
        # Non-distorted
        ax2.plot(bpm_indices, y_co_self*1e6, '-', color='blue', 
                label='Non-Distorted Closed Orbit', linewidth=3)
        for i in range(min(n_particles, y_self_avg.shape[0])):
            ax2.plot(bpm_indices, y_self_avg[i,:]*1e6, color='blue', alpha=alpha, linewidth=lw)

        # Distorted
        ax2.plot(bpm_indices, y_co_other*1e6, '--', color='red', 
                label='Distorted Closed Orbit', linewidth=3)
        for i in range(min(n_particles, y_other_avg.shape[0])):
            ax2.plot(bpm_indices, y_other_avg[i,:]*1e6, color='red', alpha=alpha, linewidth=lw)

        ax2.set_xlabel('BPM Index', fontsize=label_font)
        ax2.set_ylabel('Vertical Position [μm]', fontsize=label_font)
        ax2.grid(True)
        ax2.legend(fontsize=label_font)
        ax2.tick_params(axis='both', labelsize=tick_font)

        # Main title
        fig.suptitle(f'Closed Orbit and Betatron Oscillations (Turns {start_turn}-{end_turn})', 
                    fontsize=label_font+2, y=1.02)

        plt.tight_layout()
        plt.savefig(f"{self.figs_save_dir}/betatron_oscillations_multi_turn_{save_label}.png", 
                format='png', bbox_inches='tight')
        plt.show()
        
    def cumulative_average(self, arr):
        """Calculate the cumulative average of an array."""
        cumsum = np.cumsum(arr)
        cumavg = cumsum / np.arange(1, len(arr) + 1)
        return cumavg

    def running_average_numpy(self, data, window_size):
        data = np.asarray(data)
        if window_size <= 0:
            raise ValueError("Window size must be positive.")
        if window_size > len(data):
            raise ValueError("Window size cannot be larger than the data length.")

        # Use 'valid' mode to ensure the window fits completely within the data
        window = np.ones(window_size) / window_size
        running_avg = np.convolve(data, window, mode='valid')
        return running_avg


class SimulationRunner:
    """
    Class to run multiple synchrotron simulations based on provided configurations.
    It handles simulations both with and without quadrupole misalignments and stores
    the simulator instances for later analysis or plotting.
    """

    def __init__(self, base_configurations, common_parameters):
        """
        Initializes the SimulationRunner with base configurations and common parameters.

        Parameters:
        - base_configurations (list of dict): List of base synchrotron configurations.
        - common_parameters (dict): Dictionary containing common simulation parameters and initial condition ranges.
        """
        self.base_configurations = base_configurations
        self.common_parameters = common_parameters
        self.simulators_no_error = {}    # Dict to store simulators without errors
        self.simulators_with_error = {}  # Dict to store simulators with errors
        self.initial_states = None

    def generate_init_states(self, mean, std_dev, size=1):
        samples = np.random.normal(loc=mean, scale=std_dev, size=size)
        return samples

    def generate_init_pos_radius(self, num_particles=1000, R=0.01):
        """
        Generate random particle positions within a circle of radius R in meters.
        """
        
        # Generate random radius values (r) uniformly within the circle
        r = np.sqrt(np.random.uniform(0, R**2, num_particles))  # Uniform in area

        # Generate random angles (theta) uniformly between 0 and 2*pi
        theta = np.random.uniform(0, 2 * np.pi, num_particles)

        # Calculate x0 and y0 for each particle
        x0 = r * np.cos(theta)
        y0 = r * np.sin(theta)

        return x0, y0

    def run_configurations(self, initial_states=None, draw_plots=True, verbose=True):
        """
        Runs simulations for each base configuration both without and with quadrupole errors,
        and generates comparison plots if errors are introduced.
        """
        for base_config in self.base_configurations:
            # --- Merge Base Config with Common Parameters ---
            # Base config overrides common parameters if keys overlap
            merged_config = {**self.common_parameters, **base_config}

            config_name = merged_config.get('config_name', 'Unnamed Configuration')
            if verbose:
                print("\n" + "="*80)
                print(f"Running {config_name}:")
                print(f"  n_FODO={merged_config['n_FODO']}, design_radius={merged_config['design_radius']}m, "
                    f"f={merged_config['f']}m, L_quad={merged_config['L_quad']}m, "
                    f"L_straight={merged_config['L_straight']}m")
                print("="*80)

            try:
                # --- Simulation Without Error ---
                config_no_error = merged_config.copy()
                config_no_error['config_name'] = f"{config_name} - No Error"
                config_no_error['quad_errors'] = None  # Ensure no error is present
                config_no_error['quad_tilt_errors'] = None  # Ensure no error is present
                config_no_error['dipole_tilt_errors'] = None  # Ensure no error is present

                # Initialize simulator without error, including use_gpu
                simulator_no_error = SynchrotronSimulator(
                    design_radius=config_no_error['design_radius'],
                    G=config_no_error['G'],
                    f=config_no_error['f'],
                    use_thin_lens=config_no_error['use_thin_lens'],
                    L_quad=config_no_error['L_quad'],
                    L_straight=config_no_error['L_straight'],
                    p=config_no_error['p'],
                    q=config_no_error['q'],
                    n_turns=config_no_error['n_turns'],
                    total_dipole_bending_angle=config_no_error['total_dipole_bending_angle'],
                    num_particles=config_no_error['num_particles'],
                    n_FODO=config_no_error['n_FODO'],
                    L_dipole=config_no_error['L_dipole'],
                    n_Dipoles=config_no_error['n_Dipoles'],
                    mag_field_range=merged_config.get('mag_field_range', (0.5, 2.0)),
                    dipole_length_range=merged_config.get('dipole_length_range', (0.5, 5.0)),
                    horizontal_tune_range=merged_config.get('horizontal_tune_range', (0.2, 0.8)),
                    vertical_tune_range=merged_config.get('vertical_tune_range', (0.2, 0.8)),
                    use_gpu=merged_config.get('use_gpu', False),
                    verbose=merged_config.get('verbose', False),
                    figs_save_dir=merged_config.get('figs_save_dir', 'figs'),
                )
                
                if verbose:
                    print(simulator_no_error.backend_uasge_msg)
                    simulator_no_error.describe()

                # initial_states can be passed to the `run_configurations()`
                # for re-running the simulation with the same particles initial states.
                if initial_states is None:
                    # Extract initial condition ranges
                    x0_mean, x0_std = merged_config['x0_mean_std']
                    xp0_mean, xp0_std = merged_config['xp0_mean_std']
                    y0_mean, y0_std = merged_config['y0_mean_std']
                    yp0_mean, yp0_std = merged_config['yp0_mean_std']
                    
                    
                    # Generate initial_states based on the extracted ranges
                    if merged_config['particles_sampling_method'] in ['normal', 'circle_with_radius']:
                        if merged_config['particles_sampling_method'] == 'normal':
                            x0s = self.generate_init_states(x0_mean, x0_std, size=merged_config['num_particles'])
                            y0s = self.generate_init_states(y0_mean, y0_std, size=merged_config['num_particles'])
                        
                        elif merged_config['particles_sampling_method'] == 'circle_with_radius':
                            radius = merged_config['sampling_circle_radius']
                            x0s, y0s = self.generate_init_pos_radius(num_particles=merged_config['num_particles'])
                        
                        # X' Y' angles in radians                
                        if xp0_mean == xp0_std == 0.0:
                            xp0s = np.zeros(merged_config['num_particles'])
                        else:
                            xp0s = self.generate_init_states(xp0_mean, xp0_std, size=merged_config['num_particles'])
                        
                        if yp0_mean == yp0_std == 0.0:
                            yp0s = np.zeros(merged_config['num_particles'])
                        else:
                            yp0s = self.generate_init_states(yp0_mean, yp0_std, size=merged_config['num_particles'])
                        
                        mean_x0s = np.mean(x0s)
                        mean_y0s = np.mean(y0s)
                        
                        initial_states = np.stack([x0s, xp0s, y0s, yp0s], axis=1)
                        self.initial_states = initial_states
                    
                        if verbose:
                            print(f"init X and Y sampling method: {merged_config['particles_sampling_method']}")
                            print(f"init states mean_x0s={mean_x0s}, mean_y0s={mean_y0s}")
                            print("initial_states = ", initial_states)
                    
                    elif merged_config['particles_sampling_method'] == 'from_twiss_params':
                        alpha_x, beta_x, epsilon_x, alpha_y, beta_y, epsilon_y = simulator_no_error.compute_twiss_parameters()
                        initial_states = simulator_no_error.generate_initial_states_from_twiss(alpha_x, beta_x, epsilon_x, alpha_y, beta_y, epsilon_y, merged_config['num_particles'])                        
                        self.initial_states = initial_states
                else:
                    self.initial_states = initial_states

                if verbose:
                    print("initial_states = ", initial_states)

                # Simulate without errors
                simulator_no_error.simulate(initial_states)

                # Store the simulator instance
                self.simulators_no_error[config_no_error['config_name']] = simulator_no_error

                if verbose:
                    print("==========" * 10)

                # --- Simulation With Error ---
                quad_errors = merged_config.get('quad_errors', None)
                dipole_tilt_errors = merged_config.get('dipole_tilt_errors', None)
                quad_tilt_errors = merged_config.get('quad_tilt_errors', None)
                
                declared_errors_count = np.array([
                    quad_errors is not None,
                    dipole_tilt_errors is not None,
                    quad_tilt_errors is not None,
                ]).astype(int).sum()
                
                if self.common_parameters['reject_multiple_error_types'] and declared_errors_count > 1:
                    print("[Error] multiple error types are not supported")
                    raise Exception("Confguration contains multiple error types"
                                    "Choose only one type of errors [`quad_errors` | `dipole_tilt_errors` | `quad_tilt_errors`]")
                
                else:
                    config_with_error = merged_config.copy()
                    config_with_error['config_name'] = f"{config_name} - With Error"

                    if verbose:
                        print("\n" + "="*80)
                        print(f"Running {config_with_error['config_name']}:")
                        print(f"  n_FODO={config_with_error['n_FODO']}, design_radius={config_with_error['design_radius']}m, "
                            f"f={config_with_error['f']}m, L_quad={config_with_error['L_quad']}m, "
                            f"L_straight={config_with_error['L_straight']}m")
                        print("="*80)

                    # Initialize simulator with error, including use_gpu
                    simulator_with_error = SynchrotronSimulator(
                        design_radius=config_with_error['design_radius'],
                        G=config_with_error['G'],
                        f=config_with_error['f'],
                        use_thin_lens=config_with_error['use_thin_lens'],
                        L_quad=config_with_error['L_quad'],
                        L_straight=config_with_error['L_straight'],
                        p=config_with_error['p'],
                        q=config_with_error['q'],
                        n_turns=config_with_error['n_turns'],
                        total_dipole_bending_angle=config_with_error['total_dipole_bending_angle'],
                        num_particles=config_with_error['num_particles'],
                        n_FODO=config_with_error['n_FODO'],
                        L_dipole=config_with_error['L_dipole'],
                        n_Dipoles=config_with_error['n_Dipoles'],
                        mag_field_range=merged_config.get('mag_field_range', (0.5, 2.0)),
                        dipole_length_range=merged_config.get('dipole_length_range', (0.5, 5.0)),
                        horizontal_tune_range=merged_config.get('horizontal_tune_range', (0.2, 0.8)),
                        vertical_tune_range=merged_config.get('vertical_tune_range', (0.2, 0.8)),
                        use_gpu=merged_config.get('use_gpu', False),
                        verbose=merged_config.get('verbose', False),
                        figs_save_dir=merged_config.get('figs_save_dir', 'figs'),
                    )

                    if quad_errors:
                        # Introduce quadrupole misalignment errors
                        for quad_error in quad_errors:
                            simulator_with_error.set_quad_error(
                                FODO_index=quad_error['FODO_index'],
                                quad_type=quad_error['quad_type'],
                                delta=quad_error['delta'],
                                plane=quad_error['plane']
                            )
                    
                    if dipole_tilt_errors:
                        for dtilt_err_dict in base_config.get('dipole_tilt_errors', []):
                            simulator_with_error.set_dipole_tilt_error(
                                dtilt_err_dict['FODO_index'],
                                dtilt_err_dict['dipole_index'],
                                dtilt_err_dict['tilt_angle']
                            )
                    
                    if quad_tilt_errors:
                        for qtilt_err_dict in base_config.get('quad_tilt_errors', []):
                            simulator_with_error.set_quadrupole_tilt_error(
                                qtilt_err_dict['FODO_index'],
                                qtilt_err_dict['quad_type'],
                                qtilt_err_dict['tilt_angle']
                            )

                    # Rebuild lattice and recompute tunes after introducing errors
                    simulator_with_error.build_lattice()
                    simulator_with_error.compute_tunes()

                    if verbose:
                        simulator_with_error.describe()

                    # Simulate with errors
                    simulator_with_error.simulate(initial_states)

                    # Store the simulator instance
                    self.simulators_with_error[config_with_error['config_name']] = simulator_with_error

                    if draw_plots:
                        # --- Comparison Plots ---
                        print("\nGenerating Comparison Plots...")
                        # Specify the BPM (FODO cell) index you want to plot
                        cell_idx = 0  # Change this index as needed
                        simulator_no_error.plot_bpm_comparison_last_images(simulator_no_error, simulator_with_error, cell_idx=cell_idx, particles='all_mean')
                        viz_start_idx = simulator_no_error.n_turns - 100
                        viz_end_idx = simulator_no_error.n_turns
                        simulator_no_error.plot_comparison(simulator_with_error, cell_idx=cell_idx, viz_start_idx=viz_end_idx - 100,
                                             viz_end_idx=viz_end_idx, save_label="sim_test", window_size=50, plot_all=True, extra_title="All BPMs")
                        print("Comparison Plots Generated.\n")

            except ValueError as ve:
                if verbose:
                    print(f"Configuration '{config_name}' is infeasible: {ve}")
                    print("Skipping this configuration.\n")
                    stack_trace = traceback.format_exc()
                    print(stack_trace)

            except RuntimeError as re:
                print(f"Runtime error in configuration '{config_name}' with No Error simulation: {re}")
                print("Skipping this configuration.\n")
                stack_trace = traceback.format_exc()
                print(stack_trace)

            except Exception as e:
                print(f"An unexpected error occurred in configuration '{config_name}': {e}")
                print("Skipping this configuration.\n")
                stack_trace = traceback.format_exc()
                print(stack_trace)


def test_cpu_gpu_consistency():
    pass

# Run the test
test_cpu_gpu_consistency()

