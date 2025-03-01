# Lattice Correction Net repo

This project is part of the requirements for completing the master's thesis.

**Istinye University**  
**Institute of Graduate Education**  
**Master's Thesis**  
**Advisor: Assoc. Prof. Selçuk Hacıömeroğlu**  
**Department of Computer Engineering & Department of Physics**  

## Thesis Title: Machine Learning-based Determination of Quadrupole Misalignments in Particle Accelerators

### Abstract
Particle accelerators rely on precisely tuned magnets placed on the lattice to confine beams at high energies and maintain the design trajectory. Achieving such precision mechanically is very hard, and small magnet misalignments or tilts can degrade beam performance and stability. Detecting and correcting these inevitable faults in the lattice design is essential for preserving beam quality. In this work, we propose a machine-learning framework for lattice fault detection and correction in a synchrotron storage ring. Two neural network models were trained on simulated Beam Position Monitor (BPM) signals---a fully connected network (FCNN) and a Long Short-Term Memory (LSTM) network---to predict vertical quadrupole misalignments. We generate the data from our newly developed Python-based simulator for simulating beam dynamics. To evaluate robustness, generalization of the models, and data scalability, we benchmark our models through four modes; BPM noise and tilt noise tolerance evaluation; cross-validation; and accumulated training. The two networks effectively restored the beam’s vertical orbits to near-baseline conditions post-correction. Under realistic misalignment levels and quadrupole tilt angles (mean of 0\,µm, standard deviation of 50\,µm for vertical misalignments; mean of 0\,mrad, standard deviation of 10\,mrad for tilt angles), the proposed machine learning models achieve low mean absolute errors. Specifically, the model trained with simulated vertical misalignment errors achieves a Mean Absolute Error (MAE) of 0.57\,µm, while the model trained with simulated vertical misalignment and tilt errors achieves an MAE of 0.04\,µm. These results demonstrate the potential of ML-driven fault diagnosis and correction as a fast and accurate approach for modern accelerators, enabling stable, high-quality beams. 


## Overview

- **Simulation**: Uses a GPU-enabled synchrotron simulator to generate 4D datasets with possible lattice errors.
- **Data Processing**: Prepares and scales simulation data for neural network training.
- **Model Training**: Implements various architectures (LSTM, FCNN) for error prediction.
- **Evaluation & Benchmarking**: Provides evaluation routines and visualizations to assess model performance.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/aRibra/lattice_error_net.git
   ```
2. Create and activate a virtual environment:
   ```
   python -m venv lat
   source lat/bin/activate
   ```
3. Install required packages:
   ```
   pip install -r requirements.txt
   ```
## Key Files

- `main.py`: Entry point for generating data, training, evaluation, and benchmarking.
- `synchrotron_simulator_gpu_Dataset_4D.py`: Manages the CPU/GPU-accelerated dataset simulation.
- `sim_config.py`: Contains configuration parameters, including target experiment folder and all simulation-related and data generation parameters.
- `eval.py`: Model evaluation routines.
- `automate_dataset_collection.py`: Automates simulation and dataset collection.
- `cross_validation.py`: Provides K-Fold cross-validation benchmark.
- `accumulated_training.py`: Implements training on accumulated datasets.

## Usage

The `main.py` script lets you generate data, train a model, evaluate it, and run benchmarks. Control its behavior with command-line arguments.

### Examples

- **Generate Data & Train:**

  ```bash
  python main.py --generate-data --prepare-data --train --num-epochs 900 --n-simulations 1000 --batch-size 16 --test-size 0.10
  ```

- **Load Data & Evaluate:**

  ```bash
  python main.py --load-data --data-dir /path/to/data --load-checkpoint --evaluate --benchmark --benchmark-type quad_tilt
  ```

### Key Options

- `--generate-data`: Generate simulation data.
- `--load-data`: Load data from a directory (use with `--data-dir`).
- `--prepare-data`: Prepare data for training/inference.
- `--train`: Train a new model.
- `--load-checkpoint`: Load an existing model checkpoint.
- `--evaluate`: Evaluate the model.
- `--benchmark`: Run benchmarks.
- `--model-arch`: Set model architecture (`LSTM`, `SIMPLE_FULLY_CONNECTED`).
- `--num-epochs`: Number of training epochs.
- `--batch-size`: Batch size.
- `--benchmark-type`: Benchmark type (`bpm` or `quad_tilt`).

