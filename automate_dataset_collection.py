# automate_dataset_collection.py

import os
import torch
import numpy as np
from tqdm import tqdm
import traceback
from sklearn.preprocessing import MinMaxScaler

from synchrotron_simulator_gpu_Dataset_4D import SynchrotronSimulator, SimulationRunner, QuadrupoleMisAlignError, QuadrupoleTiltError, DipoleTiltError


class SimulationDataset:
    def __init__(self,
                 merged_config,
                 bpm_readings_no_error, 
                 bpm_readings_with_error, 
                 bpm_positions, 
                 quadrupole_errors, 
                 quadrupole_tilt_errors, 
                 dipole_tilt_errors,
                 lattice_reference,
                 apply_avg=False):
        """
        Initialize the SimulationDataset.

        Parameters:
            merged_config (dict): Merged config
            bpm_readings_no_error (dict): BPM readings from simulation without errors.
            bpm_readings_with_error (dict): BPM readings from simulation with errors.
            bpm_positions (list): Positions of the BPMs.
            quadrupole_errors (list): List of QuadrupoleMisAlignError instances.
            quadrupole_tilt_errors (list): List of QuadrupoleTiltError instances.
            dipole_tilt_errors (list): List of DipoleTiltError instances.
            lattice_reference (LatticeReference): Reference to the lattice design.
            apply_avg (bool): Whether to apply running average to the BPM data.
        """
        self.merged_config = merged_config
        self.bpm_readings_no_error = bpm_readings_no_error
        self.bpm_readings_with_error = bpm_readings_with_error
        self.bpm_positions = bpm_positions
        self.quadrupole_errors = quadrupole_errors
        self.quadrupole_tilt_errors = quadrupole_tilt_errors
        self.dipole_tilt_errors = dipole_tilt_errors
        self.lattice_reference = lattice_reference
        self.apply_avg = apply_avg

    def generate_data(self, start_rev, end_rev, fodo_cell_indices, planes, include_no_error_data=False):
        """
        Generate data for training the network.

        Parameters:
            start_rev (int): Starting revolution index.
            end_rev (int): Ending revolution index.
            fodo_cell_indices (list): List of FODO cell indices (BPMs) to consider.
            planes (list): ['x', 'y'], ['x'], or ['y']

        Returns:
            input_tensor (torch.Tensor): Input data for the network.
                Shape: [n_training_samples, n_turns, n_BPMs, len(planes)]
            target_tensor (torch.Tensor): Target data for the network.
                Shape: [n_training_samples, n_errors]
            error_values_quad_misalign (torch.Tensor): Quadrupole misalignment error(s).
            error_values_quad_tilt (torch.Tensor): Quadrupole tilt angle error(s).
            error_values_dipole_tilt (torch.Tensor): Dipole tilt angle error(s).
        """

        errors_list_quad_mis_align = []
        errors_list_quad_tilt_angles = []
        errors_list_dipole_tilt_angles = []
        
        errors_state = {
            'quadrupole_errors': len(self.quadrupole_errors) != 0,
            'quadrupole_tilt_errors': len(self.quadrupole_tilt_errors) != 0,
            'dipole_tilt_errors': len(self.dipole_tilt_errors) != 0,
        }
        len_defined_errors = len(np.flatnonzero(list(errors_state.values())))
        
        if self.merged_config['reject_multiple_error_types'] and len_defined_errors > 1:
            raise Exception(f"SimulationDataset/generate_data(): Defining multiple error types is rejected in the config"
                                        " in data automation phase. Please pass `reject_multiple_error_types=True`"
                                        f" errors_state: {errors_state}")

        errors_list_quad_mis_align = self.quadrupole_errors
        errors_list_quad_tilt_angles = self.quadrupole_tilt_errors
        errors_list_dipole_tilt_angles = self.dipole_tilt_errors
        
        # Validate planes
        if not set(planes).issubset({'x', 'y'}):
            raise ValueError("planes parameter must be a subset of ['x', 'y']")

        # Collect data
        readings_no_error = self.bpm_readings_no_error
        readings_with_error = self.bpm_readings_with_error

        # Initialize lists to store data
        data_no_error = []
        data_with_error = []

        # For each plane, compute the mean over particles, and select the specified BPMs and revolutions
        for plane in planes:
            readings_plane_no_error = readings_no_error[plane][:, start_rev:end_rev, fodo_cell_indices]
            readings_plane_with_error = readings_with_error[plane][:, start_rev:end_rev, fodo_cell_indices]

            # Compute mean over particles
            mean_no_error = np.mean(readings_plane_no_error, axis=0)  # Shape: [n_turns, n_BPMs]
            mean_with_error = np.mean(readings_plane_with_error, axis=0)  # Shape: [n_turns, n_BPMs]

            all_ravg_no_error = []
            all_ravg_with_error = []
            
            # Loop over BPMs
            for cell_ix in range(len(fodo_cell_indices)):
                if self.apply_avg:
                    ravg_no_error = self.running_average_numpy(mean_no_error[:, cell_ix],
                                                               window_size=end_rev - start_rev - 100)
                    ravg_with_error = self.running_average_numpy(mean_with_error[:, cell_ix],
                                                                 window_size=end_rev - start_rev - 100)
                else:
                    ravg_no_error = mean_no_error[:, cell_ix]
                    ravg_with_error = mean_with_error[:, cell_ix]
                    
                all_ravg_no_error.append(ravg_no_error[..., np.newaxis])
                all_ravg_with_error.append(ravg_with_error[..., np.newaxis])
            
            # concat data for all BPMs together
            all_ravg_no_error = np.concatenate(all_ravg_no_error, axis=-1)
            all_ravg_with_error = np.concatenate(all_ravg_with_error, axis=-1)
    
            data_no_error.append(all_ravg_no_error[..., np.newaxis])  # Shape: [n_turns, n_BPMs, 1]
            data_with_error.append(all_ravg_with_error[..., np.newaxis])  # Shape: [n_turns, n_BPMs, 1]

        # Concatenate planes if necessary
        if len(planes) > 1:
            data_no_error = np.concatenate(data_no_error, axis=-1)     # Shape: [n_turns, n_BPMs, len(planes)]
            data_with_error = np.concatenate(data_with_error, axis=-1)
        else:
            data_no_error = data_no_error[0]       # Shape: [n_turns, n_BPMs, 1]
            data_with_error = data_with_error[0]

        if include_no_error_data:
            # Stack data along a new axis to create n_training_samples dimension
            input_data = np.stack([data_with_error, data_no_error], axis=0)
        else:
            input_data = np.stack([data_with_error], axis=0)
            
        no_error_value = 0.0
        error_values_quad_misalign = []
        error_values_quad_tilt = []
        error_values_dipole_tilt = []
        
        for error in errors_list_quad_mis_align:
            if isinstance(error, QuadrupoleMisAlignError):
                error_values_quad_misalign.append(error.delta)
                
        for error in errors_list_quad_tilt_angles:
            if isinstance(error, QuadrupoleTiltError):
                error_values_quad_tilt.append(error.tilt_angle)
                
        for error in errors_list_dipole_tilt_angles:
            if isinstance(error, DipoleTiltError):
                error_values_dipole_tilt.append(error.tilt_angle)
        
        print(self.merged_config)
        
        if not self.merged_config['target_data']:
            target_error_values = []
            print("[WARNING] SimulationDataset.generate_data()/ `target_data` was not specified in the configuration. "
                  "Therefore, there will be no errors applied.")
        elif self.merged_config['target_data'] == 'quad_misalign_deltas' and errors_state['quadrupole_errors']:
            target_error_values = error_values_quad_misalign
        elif self.merged_config['target_data'] == 'quad_tilt_angles' and errors_state['quadrupole_tilt_errors']:
            target_error_values = error_values_quad_tilt
        elif self.merged_config['target_data'] == 'dipole_tilt_angles' and errors_state['dipole_tilt_errors']:
            target_error_values = error_values_dipole_tilt
        else:
            raise ValueError("This exception indicates that you selected a `target_data` type, "
                             "but did not configure any errors in the configuration. "
                             "Check `target_data` value and if there were any errors configured "
                             "for that target error type.")
        

        # Handle the case where there are no errors
        if len(target_error_values) == 0:
            target_error_values = [no_error_value]

        n_errors = len(target_error_values)
        n_training_samples = 1 if not include_no_error_data else 2
        target_data = np.zeros((n_training_samples, n_errors))
        target_data[0, :] = target_error_values
        if include_no_error_data:
            target_data[1, :] = no_error_value

        # Flatten input_data for scaling
        n_training_samples, n_turns, n_BPMs, n_features = input_data.shape
        input_data_reshaped = input_data.reshape(n_training_samples * n_turns * n_BPMs, n_features)

        # Reshape back for torch
        input_data_reshaped = input_data_reshaped.reshape(n_training_samples, n_turns, n_BPMs, n_features)

        # Convert to torch tensors
        input_tensor = torch.tensor(input_data_reshaped, dtype=torch.float32)
        target_tensor = torch.tensor(target_data, dtype=torch.float32)

        error_values_quad_misalign = torch.tensor(error_values_quad_misalign, dtype=torch.float32)[np.newaxis, ...]
        error_values_quad_tilt = torch.tensor(error_values_quad_tilt, dtype=torch.float32)[np.newaxis, ...]
        error_values_dipole_tilt = torch.tensor(error_values_dipole_tilt, dtype=torch.float32)[np.newaxis, ...]

        return (input_tensor, target_tensor,
                error_values_quad_misalign,
                error_values_quad_tilt,
                error_values_dipole_tilt)

    def inverse_transform_target(self, scaled_target):
        """Inverse transform the scaled target back to original scale."""
        return self.target_scaler.inverse_transform(scaled_target)

    def print_lattice_reference(self):
        """Print the lattice reference information."""
        self.lattice_reference.describe()

    def running_average_numpy(self, data, window_size):
        data = np.asarray(data)
        if window_size <= 0:
            raise ValueError("Window size must be positive.")
        if window_size > len(data):
            raise ValueError("Window size cannot be larger than the data length.")

        window = np.ones(window_size) / window_size
        running_avg = np.convolve(data, window, mode='valid')
        return running_avg


class DataAutomation:
    def __init__(self, base_configurations, common_parameters, n_simulations):
        """
        Initializes the DataAutomation class.

        Parameters:
            base_configurations (list): List of base configuration dictionaries.
            common_parameters (dict): Dictionary of common parameters.
            n_simulations (int): Number of simulations to run.
        """
        self.base_configurations = base_configurations
        self.common_parameters = common_parameters
        self.n_simulations = n_simulations

        self.merged_config = None
    
        quad_errors = base_configurations[0]['quad_errors']
        quad_tilt_errors = base_configurations[0]['quad_tilt_errors']
        dipole_tilt_errors = base_configurations[0]['dipole_tilt_errors']
        
        if quad_errors:
            fodo_indices_with_error = [err['FODO_index'] for err in quad_errors]
            fodo_indices_with_error_str = "".join(str(ei) for ei in fodo_indices_with_error)
        else:
            fodo_indices_with_error = []
            fodo_indices_with_error_str = ""
        
        if quad_tilt_errors:
            fodo_indices_with_quad_tilt_error = [err['FODO_index'] for err in quad_tilt_errors]
            fodo_indices_with_quad_tilt_error_str = "".join(str(ei) for ei in fodo_indices_with_quad_tilt_error)
        else:
            fodo_indices_with_quad_tilt_error = []
            fodo_indices_with_quad_tilt_error_str = ""
        
        if dipole_tilt_errors:
            fodo_indices_with_dipole_tilt_error = [err['FODO_index'] for err in dipole_tilt_errors]
            fodo_indices_with_dipole_tilt_error_str = "".join(str(ei) for ei in fodo_indices_with_dipole_tilt_error)
        else:
            fodo_indices_with_dipole_tilt_error = []
            fodo_indices_with_dipole_tilt_error_str = ""

        # For readability, label each section: 
        tag = (
            f"Sim{n_simulations}_"                     # number of total samples
            f"{common_parameters['n_turns']}turns_"    # turns per simulation
            f"{common_parameters['num_particles']}parts_"  # number of particles
            f"FODOErr-{fodo_indices_with_error_str}-"
            f"{fodo_indices_with_quad_tilt_error_str}-"
            f"{fodo_indices_with_dipole_tilt_error_str}_"  # which FODOs had errors
            f"avg{common_parameters['apply_avg']}_"    # whether averaging is applied
            f"tgt{common_parameters['target_data']}"   # which target variable is used
        )

        self.output_folder_path = self.create_tagged_folder(tag)
        
        self.tag = os.path.basename(self.output_folder_path)
        
        # Initialize data storage
        self.all_input_tensors = []
        self.all_target_tensors = []
        self.all_input_tensors_scaled = None
        self.all_target_tensors_scaled = None
        self.com_deltas_x = []
        self.com_deltas_y = []
        
        self.all_error_values_quad_misalign = []
        self.all_error_values_quad_tilt = []
        self.all_error_values_dipole_tilt = []

        # Placeholders for feasible lattice design and overridden base configuration
        self.lattice_design = None
        self.overridden_base_config = None
        
        # Data scalers
        self.data_scalers = {
            'input_scaler': MinMaxScaler(feature_range=(-1, 1)),
            'target_scaler': MinMaxScaler(feature_range=(-1, 1))
        }

    def find_feasible_lattice(self):
        """
        Finds a feasible lattice design using the provided base configurations and common parameters.
        """
        feasible_lattices = SynchrotronSimulator.find_feasible_lattices(
            base_configurations=self.base_configurations,
            common_parameters=self.common_parameters
        )

        if not feasible_lattices:
            raise Exception("No feasible lattice configurations found.")
        else:
            self.lattice_design = feasible_lattices[0]
            print(f"self.lattice_design: \n {self.lattice_design}")

    def override_base_config(self):
        """
        Overrides the base configuration with the feasible lattice design.
        """
        base_config = self.base_configurations[0].copy()
        lattice_design = self.lattice_design

        for param in lattice_design.keys():
            base_config[param] = lattice_design[param]
        base_config['config_name'] = base_config.get('config_name', 'Configuration')
        base_config['quad_errors'] = base_config.get('quad_errors', None)

        self.overridden_base_config = base_config
        
        # Set `merged_config`` 
        self.merged_config = {**self.common_parameters, **self.overridden_base_config}

    def compute_com_deltas(self, simulator_no_error, simulator_with_error, cell_idx=0):
        """
        Computes the center of mass delta X and delta Y between simulations with and without error.

        Parameters:
            simulator_no_error (SynchrotronSimulator): Simulator instance without error.
            simulator_with_error (SynchrotronSimulator): Simulator instance with error.
            cell_idx (int): BPM index (FODO cell) to consider.

        Returns:
            tuple: (delta_x, delta_y)
        """
        start_idx = self.common_parameters.get('start_rev', 0)
        end_idx = self.common_parameters.get('end_rev', simulator_no_error.n_turns)

        x_no_error = simulator_no_error.bpm_readings['x'][:, start_idx:end_idx, cell_idx].mean()
        y_no_error = simulator_no_error.bpm_readings['y'][:, start_idx:end_idx, cell_idx].mean()
        x_with_error = simulator_with_error.bpm_readings['x'][:, start_idx:end_idx, cell_idx].mean()
        y_with_error = simulator_with_error.bpm_readings['y'][:, start_idx:end_idx, cell_idx].mean()

        delta_x = x_with_error - x_no_error
        delta_y = y_with_error - y_no_error

        return delta_x, delta_y

    def run_simulations(self, skip_data_on_delta_ranges=False, include_no_error_data=False):
        """
        Runs simulations with the same design, varying only the quadrupole error delta and initial conditions.
        Collects the input and target data tensors from each simulation.
        """

        quad_misalign_delta_mean_or_min, quad_misalign_delta_std_or_max = self.common_parameters['delta_range']
        quad_tilt_mean_or_min, quad_tilt_std_or_max = self.common_parameters['quad_tilt_angle_range']
        dipole_tilt_mean_or_min, dipole_tilt_std_or_max = self.common_parameters['dipole_tilt_angle_range']
        delta_range_min, delta_range_max = self.common_parameters['com_delta_range']

        for sim_idx in tqdm(range(self.n_simulations), desc="Running Simulations"):
            base_config = self.overridden_base_config.copy()
            
            errors_cfg_state = {
                'quad_errors': base_config['quad_errors'] is not None,
                'quad_tilt_errors': base_config['quad_tilt_errors'] is not None,
                'dipole_tilt_errors': base_config['dipole_tilt_errors'] is not None,
            }
            
            len_defined_errors = len(np.flatnonzero(list(errors_cfg_state.values())))
            
            if self.common_parameters['reject_multiple_error_types'] and len_defined_errors > 1:
                raise Exception("Defining multiple error types is not currently supported "
                                "in data automation phase. Please define only one type"
                                f"The defined errors in config are: {errors_cfg_state}")
            elif len_defined_errors == 0:
                raise ValueError("None of [`quad_errors`|`quad_tilt_errors`|`dipole_tilt_errors`] "
                                 "was defined in the current configuration.")

            sampling_func = None
            if self.common_parameters['random_criterion'] == 'uniform':
                sampling_func = np.random.uniform
            elif self.common_parameters['random_criterion'] == 'normal':
                sampling_func = np.random.normal
    
            if base_config['quad_errors']:
                for quad_error_idx in range(len(base_config['quad_errors'])):
                    random_delta = sampling_func(quad_misalign_delta_mean_or_min, quad_misalign_delta_std_or_max)
                    base_config['quad_errors'][quad_error_idx]['delta'] = random_delta

            if base_config['quad_tilt_errors']:
                for quad_error_idx in range(len(base_config['quad_tilt_errors'])):
                    random_tilt_angle = sampling_func(quad_tilt_mean_or_min, quad_tilt_std_or_max)
                    base_config['quad_tilt_errors'][quad_error_idx]['tilt_angle'] = random_tilt_angle

            if base_config['dipole_tilt_errors']:
                for quad_error_idx in range(len(base_config['dipole_tilt_errors'])):
                    random_tilt_angle = sampling_func(dipole_tilt_mean_or_min, dipole_tilt_std_or_max)
                    base_config['dipole_tilt_errors'][quad_error_idx]['tilt_angle'] = random_tilt_angle

            base_configurations = [base_config]
            runner = SimulationRunner(
                base_configurations=base_configurations,
                common_parameters=self.common_parameters
            )

            try:
                runner.run_configurations(draw_plots=False, verbose=False)

                config_key_no_error = f"{base_config['config_name']} - No Error"
                config_key_with_error = f"{base_config['config_name']} - With Error"

                simulator_no_error = runner.simulators_no_error.get(config_key_no_error)
                simulator_with_error = runner.simulators_with_error.get(config_key_with_error)

                if simulator_no_error and simulator_with_error:
                    delta_x, delta_y = self.compute_com_deltas(simulator_no_error, simulator_with_error)

                    if skip_data_on_delta_ranges and \
                        not (delta_range_min <= delta_x <= delta_range_max and
                             delta_range_min <= delta_y <= delta_range_max):
                        print(f"Simulation {sim_idx}: Delta out of range (delta_x={delta_x}, "
                              f"delta_y={delta_y}), random_delta={random_delta} skipping.")
                        continue
                    else:
                        self.com_deltas_x.append(delta_x)
                        self.com_deltas_y.append(delta_y)

                        simulation_dataset = SimulationDataset(
                            merged_config=self.merged_config,
                            bpm_readings_no_error=simulator_no_error.bpm_readings,
                            bpm_readings_with_error=simulator_with_error.bpm_readings,
                            bpm_positions=simulator_no_error.bpm_positions,
                            quadrupole_errors=simulator_with_error.quad_errors,
                            quadrupole_tilt_errors=simulator_with_error.quadrupole_tilt_errors,
                            dipole_tilt_errors=simulator_with_error.dipole_tilt_errors,
                            lattice_reference=simulator_no_error.get_lattice_reference(),
                            apply_avg=self.common_parameters.get('apply_avg', False)
                        )

                        start_rev = self.common_parameters.get('start_rev', 0)
                        end_rev = self.common_parameters.get('end_rev', simulator_no_error.n_turns)
                        fodo_cell_indices = self.common_parameters.get('fodo_cell_indices',
                                                                       list(range(simulator_no_error.n_FODO)))
                        planes = self.common_parameters.get('planes', ['x', 'y'])

                        (input_tensor, target_tensor,
                         error_values_quad_misalign,
                         error_values_quad_tilt,
                         error_values_dipole_tilt) = simulation_dataset.generate_data(
                            start_rev, end_rev, fodo_cell_indices, planes,
                            include_no_error_data=include_no_error_data
                        )

                        self.all_input_tensors.append(input_tensor)
                        self.all_target_tensors.append(target_tensor)
                        self.all_error_values_quad_misalign.append(error_values_quad_misalign)
                        self.all_error_values_quad_tilt.append(error_values_quad_tilt)
                        self.all_error_values_dipole_tilt.append(error_values_dipole_tilt)

                        # Save after each successful simulation
                        self.save_data(postfix="accumulated")

                else:
                    print(f"Simulation {sim_idx} failed or missing simulators.")
            except Exception as e:
                print(f"Simulation {sim_idx} encountered an error: {e}")
                stack_trace = traceback.format_exc()
                print(stack_trace)
                continue
        
        self.save_data(postfix="final")

    def get_data_tensors(self):
        """
        Returns the concatenated input and target data tensors.

        Returns:
            tuple: (input_tensors, target_tensors, input_tensors_scaled, target_tensors_scaled)
        """
        return (self.all_input_tensors_torch,
                self.all_target_tensors_torch,
                self.all_input_tensors_scaled_torch,
                self.all_target_tensors_scaled_torch,
                self.all_error_values_quad_misalign_torch,
                self.all_error_values_quad_tilt_torch,
                self.all_error_values_dipole_tilt_torch)

    def get_error_values(self):
        """
        Returns the concatenated error values tensors.

        Returns:
            tuple: (error_values_quad_misalign, error_values_quad_tilt, error_values_dipole_tilt)
        """
        return (self.error_values_quad_misalign,
                self.error_values_quad_tilt,
                self.all_error_values_dipole_tilt)

    def run(self, skip_data_on_delta_ranges=False, include_no_error_data=False):
        """
        High-level method to run all steps: find feasible lattice, override base configuration,
        run simulations, and collect data tensors.

        Returns:
            tuple: (input_tensors, target_tensors, input_tensors_scaled, target_tensors_scaled)
        """
        self.find_feasible_lattice()
        self.override_base_config()
        self.run_simulations(skip_data_on_delta_ranges=skip_data_on_delta_ranges,
                             include_no_error_data=include_no_error_data)
        return self.get_data_tensors()

    def get_data_scalers(self):
        return self.data_scalers

    def save_data(self, postfix="accumulated"):
        """
        Saves the current state of DataAutomation, including:
          - This DataAutomation object
          - The merged_config
          - The input/target tensors (raw and scaled)
          - The data scalers
        All files are overwritten with the given 'tag' to avoid losing progress 
        when running many simulations.
        """
        
        # After the loop, concatenate all tensors
        if self.all_input_tensors and self.all_target_tensors:
            self.all_input_tensors_torch = torch.cat(self.all_input_tensors, dim=0)
            self.all_target_tensors_torch = torch.cat(self.all_target_tensors, dim=0)

            # Remove entire data samples (rows) if any NaN is present in that sample.
            n_samples, n_turns, n_BPMs, n_features = self.all_input_tensors_torch.shape
            input_data_flat = self.all_input_tensors_torch.view(n_samples, -1)   # shape (n_samples, n_turns*n_BPMs*n_features)
            nan_mask = torch.isnan(input_data_flat).any(dim=1)            # True if any feature is NaN in that sample
            keep_mask = ~nan_mask                                         # we keep only rows with no NaNs

            print(f"Removing {nan_mask.sum().item()} samples containing NaNs.")
            self.all_input_tensors_torch = self.all_input_tensors_torch[keep_mask]
            self.all_target_tensors_torch = self.all_target_tensors_torch[keep_mask]


            n_samples, n_turns, n_BPMs, n_features = self.all_input_tensors_torch.shape
            input_data_reshaped = self.all_input_tensors_torch.reshape(n_samples * n_turns * n_BPMs, n_features)

            input_data_scaled = self.data_scalers['input_scaler'].fit_transform(input_data_reshaped)
            self.all_input_tensors_scaled_torch = input_data_scaled.reshape(n_samples, n_turns, n_BPMs, n_features)
            self.all_input_tensors_scaled_torch = torch.tensor(self.all_input_tensors_scaled_torch, dtype=torch.float32)

            if len(self.all_target_tensors_torch.shape) > 2:
                n_samples, n_targets = self.all_target_tensors_torch.shape
                target_data_reshaped = self.all_target_tensors_torch.reshape(n_samples, n_targets)
            else:
                target_data_reshaped = self.all_target_tensors_torch

            self.all_target_tensors_scaled_torch = self.data_scalers['target_scaler'].fit_transform(target_data_reshaped)
            self.all_target_tensors_scaled_torch = torch.tensor(self.all_target_tensors_scaled_torch, dtype=torch.float32)

            self.all_error_values_quad_misalign_torch = torch.cat(self.all_error_values_quad_misalign, dim=0)
            self.all_error_values_quad_tilt_torch = torch.cat(self.all_error_values_quad_tilt, dim=0)
            self.all_error_values_dipole_tilt_torch = torch.cat(self.all_error_values_dipole_tilt, dim=0)
        else:
            self.all_input_tensors_torch = None
            self.all_target_tensors_torch = None
            self.all_error_values_quad_misalign = None
            self.all_error_values_quad_tilt = None
            self.all_error_values_dipole_tilt = None

        torch.save(self, f"{self.output_folder_path}/data_automation-{postfix}.pt")
        torch.save(self.merged_config, f"{self.output_folder_path}/merged_config-{postfix}.pt")
        torch.save(self.all_input_tensors_torch, f"{self.output_folder_path}/input_tensors-{postfix}.pt")
        torch.save(self.all_target_tensors_torch, f"{self.output_folder_path}/target_tensors-{postfix}.pt")
        torch.save(self.all_input_tensors_scaled_torch, f"{self.output_folder_path}/input_tensors_scaled-{postfix}.pt")
        torch.save(self.all_target_tensors_scaled_torch, f"{self.output_folder_path}/target_tensors_scaled-{postfix}.pt")
        
        torch.save(self.all_error_values_quad_misalign_torch, f"{self.output_folder_path}/all_error_values_quad_misalign-{postfix}.pt")
        torch.save(self.all_error_values_quad_tilt_torch, f"{self.output_folder_path}/all_error_values_quad_tilt-{postfix}.pt")
        torch.save(self.all_error_values_dipole_tilt_torch, f"{self.output_folder_path}/all_error_values_dipole_tilt-{postfix}.pt")
        
        torch.save(self.data_scalers, f"{self.output_folder_path}/dataset_scalers-{postfix}.pt")

        print(f"[save_data] Saved `{postfix}` checkpoint with tag='{self.tag}'")
        
    def create_tagged_folder(self, tag, base_dir="data"):
        """
        Creates a folder in `base_dir` with a pattern like <tag>_1, <tag>_2, etc.
        
        - If no folders named <tag>_* exist, creates <tag>_1.
        - If some exist, finds the highest numeric suffix and uses the next integer.
        - Returns the full path to the newly created folder.
        """
        import os
        import re

        # Ensure the base directory exists
        os.makedirs(base_dir, exist_ok=True)
        
        # Gather subdirectories that start with `tag` within base_dir
        existing_dirs = []
        for item in os.listdir(base_dir):
            full_path = os.path.join(base_dir, item)
            if os.path.isdir(full_path) and item.startswith(tag):
                existing_dirs.append(item)
        
        # Identify numeric suffixes in the format <tag>_<number>
        suffix_vals = []
        pattern = rf'^{re.escape(tag)}_(\d+)$'
        for d in existing_dirs:
            match = re.match(pattern, d)
            if match:
                suffix_vals.append(int(match.group(1)))
        
        # Determine the suffix to use
        if suffix_vals:
            next_suffix = max(suffix_vals) + 1
        else:
            next_suffix = 1

        # Build the final folder name and path
        folder_name = f"{tag}_{next_suffix}"
        folder_path = os.path.join(base_dir, folder_name)

        # Create the folder
        os.makedirs(folder_path, exist_ok=True)
        
        return folder_path
