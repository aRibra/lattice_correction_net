# net.py

import os
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from visualization import plot_training_results, plot_pinn_losses
from sim_config import SAVE_DIR_TRAINING
from datetime import datetime
from utils import load_config, deserialize_minmax_scaler
from constants import Constants as C
from synchrotron_simulator_gpu_Dataset_4D import SynchrotronSimulator


class QuadErrorCorrectionLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2,
        output_size=1,
        is_bidirectional=False,
    ):
        super(QuadErrorCorrectionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.is_bidirectional = is_bidirectional

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=is_bidirectional,
        )

        # Fully connected layer to map the hidden state to the output
        self.hidden_size_mult = 1
        if self.is_bidirectional:
            self.hidden_size_mult = 2
        self.fc = nn.Linear(hidden_size * self.hidden_size_mult, output_size)

        # Activation function
        # self.tanh = nn.Tanh()

    def forward(self, x):
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(
            self.num_layers * self.hidden_size_mult, x.size(0), self.hidden_size
        ).to(x.device)  # Hidden state
        c0 = torch.zeros(
            self.num_layers * self.hidden_size_mult, x.size(0), self.hidden_size
        ).to(x.device)  # Cell state

        # Forward propagate the LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_length, hidden_size)

        # Use the last time step's output for prediction
        out = out[:, -1, :]  # (batch_size, hidden_size)

        # Pass through the fully connected layer
        out = self.fc(out)  # (batch_size, output_size)

        # out = self.tanh(out)
        return out


class SimpleFullyConnectedNetwork(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=3,
        nb_heads=None,
        add_batch_norm=False,
        add_dropout=False,
        down_s_factor=1,
        act="relu",
    ):
        super(SimpleFullyConnectedNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.nb_heads = nb_heads
        self.act = act

        # Base fully connected layers
        self.fc = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())

        # Downsample factor
        self.down_s_factor = down_s_factor

        self.add_batch_norm = add_batch_norm
        self.add_dropout = add_dropout

        for nl in range(1, num_layers + 1):
            if nl == 1:
                hdn_siz = self.hidden_size
            else:
                hdn_siz = self.hidden_size // 2 ** ((nl - 1) * down_s_factor)
            out_size = self.hidden_size // 2 ** (nl * down_s_factor)

            self.fc.append(nn.Linear(hdn_siz, out_size))

            if act == "relu":
                self.fc.append(nn.ReLU())
            elif act == "elu":
                self.fc.append(nn.ELU(alpha=1.0))

            if self.add_batch_norm:
                self.fc.append(nn.BatchNorm1d(out_size))

            if self.add_dropout:
                self.fc.append(nn.Dropout(p=0.05))

        if self.nb_heads is None:
            # no heads? add the final output layer directly
            self.fc.add_module(
                "output_layer",
                nn.Linear(
                    hidden_size // 2 ** (self.num_layers * down_s_factor),
                    self.output_size,
                ),
            )
        else:
            # ModuleDict for storing multiple head layers
            self.heads = nn.ModuleDict(
                {
                    f"head_{ixh}": nn.Linear(
                        hidden_size // 2 ** (self.num_layers * down_s_factor), 1
                    )
                    for ixh in range(self.nb_heads)
                }
            )

        # Activation functions (if needed)
        # self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        out = self.fc(x)

        if self.nb_heads is not None:
            heads_outs = []
            for head_name, head_fc in self.heads.items():
                out_head = head_fc(out)
                heads_outs.append(out_head)

            # Concatenate along the feature dimension (usually dim=1)
            out = torch.cat(heads_outs, dim=1)

        # Apply activation (if needed)
        # out = self.tanh(out)

        return out


class SimpleCNN(nn.Module):
    def __init__(
        self,
        input_channels=2,
        hidden_size=32,
        output_size=1,
        num_layers=3,
        nb_heads=None,
        add_batch_norm=False,
        add_dropout=False,
        down_s_factor=1,
        act="relu",
        target_height=12,
        target_width=4,
    ):
        """
        Initialize the SimpleCNN with Adaptive Pooling.

        Args:
            input_channels (int): Number of input channels. Default is 2 for [x, y] planes.
            hidden_size (int): Base number of output channels for the first conv layer.
            output_size (int): Size of the output layer.
            num_layers (int): Number of convolutional layers.
            nb_heads (int or None): Number of output heads. If None, a single head is used.
            add_batch_norm (bool): If True, adds BatchNorm2d after conv layers.
            add_dropout (bool): If True, adds Dropout2d after activation functions.
            down_s_factor (int): Downsampling factor to reduce spatial dimensions.
            act (str): Activation function to use ('relu' or 'elu').
            target_height (int): Target height after adaptive pooling.
            target_width (int): Target width after adaptive pooling.
        """
        super(SimpleCNN, self).__init__()

        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.nb_heads = nb_heads
        self.down_s_factor = down_s_factor
        self.add_batch_norm = add_batch_norm
        self.add_dropout = add_dropout
        self.act = act
        self.target_height = target_height
        self.target_width = target_width

        # Define activation function
        if act.lower() == "relu":
            activation_fn = nn.ReLU()
        elif act.lower() == "elu":
            activation_fn = nn.ELU(alpha=1.0)
        else:
            raise ValueError(f"Unsupported activation function: {act}")

        # Initialize convolutional layers
        self.conv_layers = nn.ModuleList()
        current_in_channels = input_channels
        current_out_channels = hidden_size

        for layer in range(1, num_layers + 1):
            # Calculate out_channels based on down_s_factor
            if layer == 1:
                out_channels = hidden_size
            else:
                out_channels = hidden_size // (2 ** ((layer - 1) * down_s_factor))
                out_channels = max(1, out_channels)  # Ensure at least 1 channel

            # Append Conv2d layer
            conv = nn.Conv2d(
                in_channels=current_in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            )
            self.conv_layers.append(conv)

            # Optional BatchNorm2d
            if add_batch_norm:
                bn = nn.BatchNorm2d(out_channels)
                self.conv_layers.append(bn)

            # Append activation function
            self.conv_layers.append(activation_fn)

            # Optional Dropout2d
            if add_dropout:
                dropout = nn.Dropout2d(p=0.05)
                self.conv_layers.append(dropout)

            # Append MaxPool2d for downsampling
            if down_s_factor > 0:
                pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)
                self.conv_layers.append(pool)

            # Update channels for next layer
            current_in_channels = out_channels

        # Adaptive Pooling to fix spatial dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (self.target_height, self.target_width)
        )

        # Calculate flatten_size
        self.flatten_size = current_in_channels * self.target_height * self.target_width
        print(
            f"Flatten size after adaptive pooling: {self.flatten_size} "
            f"(Channels: {current_in_channels}, Height: {self.target_height}, Width: {self.target_width})"
        )

        # Fully connected layers
        fcn_hidden_size = hidden_size // 2
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, fcn_hidden_size), activation_fn
        )

        if add_batch_norm:
            self.fc.add_module("bn_fc", nn.BatchNorm1d(fcn_hidden_size))

        if add_dropout:
            self.fc.add_module("dropout_fc", nn.Dropout(p=0.05))

        # Output layers
        if self.nb_heads is None:
            # Single output head
            self.fc.add_module(
                "output_layer", nn.Linear(fcn_hidden_size, self.output_size)
            )
        else:
            # Multiple output heads
            self.heads = nn.ModuleDict(
                {
                    f"head_{ixh}": nn.Linear(fcn_hidden_size, 1)
                    for ixh in range(self.nb_heads)
                }
            )

    def forward(self, x):
        """
        Forward pass of the CNN with Adaptive Pooling.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size) or (batch_size, nb_heads)
        """
        # Pass through convolutional layers
        for layer in self.conv_layers:
            x = layer(x)

        # Apply adaptive pooling
        x = self.adaptive_pool(x)

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Pass through fully connected layers
        out = self.fc(x)

        # Handle multiple heads
        if self.nb_heads is not None:
            heads_out = []
            for head_name, head_layer in self.heads.items():
                head_out = head_layer(out)
                heads_out.append(head_out)
            # Concatenate all head outputs
            out = torch.cat(heads_out, dim=1)

        return out


def build_model_from_train_dir(model_train_dir, device):
    json_cfg_path = os.path.join(model_train_dir, "config.json")
    chkpnt_path = os.path.join(model_train_dir, "best.pt")
    config = load_config(json_cfg_path)
    data_shapes = config["data_shapes"]
    model_type = config["model_type"]
    data_sub_cfg = config["data_sub_cfg"]

    model = build_model(model_type, data_shapes, device)
    model = load_from_checkpoint(model, chkpnt_path)

    return model, data_sub_cfg


def build_model(model_type, data_shapes, device, **kwargs):
    """
    Build a PyTorch model based on the specified model type.
        model_type (str): The type of model to build. Supported types are C.NET_ARCH_LSTM, C.NET_ARCH_SIMPLE_FULLY_CONNECTED, and C.NET_ARCH_SIMPLE_CNN.
        data_shapes (dict): Dictionary containing the shapes of the input and target tensors.
        device (torch.device): The device (CPU or GPU) to run the model on.
        **kwargs: Additional keyword arguments for model-specific parameters.
    Returns:
        torch.nn.Module: The constructed PyTorch model.
    Raises:
        ValueError: If an unsupported model type is provided.
    """

    _, n_turns, input_size = data_shapes["inputs_shape"]
    _, n_errors = data_shapes["targets_shape"]
    _, n_turns, n_BPMs, n_planes = data_shapes["raw_input_tensors_shape"]

    if model_type == C.NET_ARCH_LSTM:
        # LSTM model
        lstm_hidden_size = 256
        num_layers = 1
        is_bidirectional = True

        if "hidden_size" in kwargs:
            lstm_hidden_size = kwargs["hidden_size"]
        if "num_layers" in kwargs:
            num_layers = kwargs["num_layers"]
        if "is_bidirectional" in kwargs:
            is_bidirectional = kwargs["is_bidirectional"]

        model = QuadErrorCorrectionLSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            output_size=n_errors,
            is_bidirectional=is_bidirectional,
        )

    elif model_type == C.NET_ARCH_SIMPLE_FULLY_CONNECTED:
        # Simple FCNN model
        fnn_hidden_size = 32
        nb_heads = None  # output_size
        num_layers = 1
        add_dropout = False
        add_batch_norm = False
        down_s_factor = 1

        if "hidden_size" in kwargs:
            fnn_hidden_size = kwargs["hidden_size"]
        if "num_layers" in kwargs:
            num_layers = kwargs["num_layers"]
        if "nb_heads" in kwargs:
            nb_heads = kwargs["nb_heads"]
        if "add_batch_norm" in kwargs:
            add_batch_norm = kwargs["add_batch_norm"]
        if "add_dropout" in kwargs:
            add_dropout = kwargs["add_dropout"]
        if "down_s_factor" in kwargs:
            down_s_factor = kwargs["down_s_factor"]

        model = SimpleFullyConnectedNetwork(
            input_size=n_turns * input_size,
            hidden_size=fnn_hidden_size,
            output_size=n_errors,
            num_layers=num_layers,
            nb_heads=nb_heads,
            add_batch_norm=add_batch_norm,
            add_dropout=add_dropout,
            down_s_factor=down_s_factor,
        )

    elif model_type == C.NET_ARCH_SIMPLE_CNN:
        # Simple CNN model
        cnn_hidden_size = 128
        num_layers = 1
        nb_heads = None
        down_s_factor = 1
        act = "relu"
        target_height = 24
        target_width = 8

        if "hidden_size" in kwargs:
            cnn_hidden_size = kwargs["hidden_size"]
        if "num_layers" in kwargs:
            num_layers = kwargs["num_layers"]
        if "nb_heads" in kwargs:
            nb_heads = kwargs["nb_heads"]
        if "down_s_factor" in kwargs:
            down_s_factor = kwargs["down_s_factor"]
        if "act" in kwargs:
            act = kwargs["act"]
        if "target_height" in kwargs:
            target_height = kwargs["target_height"]
        if "target_width" in kwargs:
            target_width = kwargs["target_width"]

        model = SimpleCNN(
            input_channels=n_planes,
            hidden_size=cnn_hidden_size,
            output_size=n_errors,
            num_layers=num_layers,
            nb_heads=None,  # Set to desired number of heads or None
            add_batch_norm=True,
            add_dropout=False,
            down_s_factor=down_s_factor,  # downsampling factor
            act=act,  # or 'elu'
            target_height=target_height,
            target_width=target_width,
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    setattr(model, "data_shapes", data_shapes)

    model.to(device)

    print("model number_of_parameters: ", number_of_parameters(model))

    return model


def load_from_checkpoint(model, model_load_path):
    # Load model from checkpoint
    chkpnt_state_dict = torch.load(model_load_path, weights_only=True)
    model.load_state_dict(chkpnt_state_dict)
    print("Checkpoint Loaded.")
    return model


def save_model(model, model_arch, save_path, tag):
    # tag: mix_quad_error_correction_
    model_save_path = f"{save_path}/{tag}_{model_arch}.pth"

    # Save the model's state_dict
    torch.save(model.state_dict(), model_save_path)

    print(f"Model saved to {model_save_path}")


def number_of_parameters(model):
    """
    Calculate the number of trainable parameters in the given model.
    Args:
        model (torch.nn.Module): The PyTorch model to analyze.
    Returns:
        int: The number of trainable parameters in the model.
    """
    nb_trainable_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    nb_all_model_params = 0
    for p in model.parameters():
        nb_all_model_params += p.numel()
    print("nb_all_model_params = ", nb_all_model_params)
    print("nb_trainable_parameters = ", nb_trainable_parameters)

    return nb_all_model_params, nb_trainable_parameters


def compute_mae(outputs, targets, aggregate="mean"):
    absolute_errors = torch.abs(outputs - targets)  # Shape: (BATCH_SIZE, 6)

    # MAE for each output dimension
    mae_per_output = torch.mean(absolute_errors, dim=0)  # Shape: (nb_outputs,)

    if aggregate == "sum":
        return mae_per_output.sum()
    elif aggregate == "mean":
        return mae_per_output.mean()
    elif aggregate == "none":
        return mae_per_output
    else:
        raise ValueError("Invalid aggregation method. Choose 'sum', 'mean', or 'none'.")
    

def get_scheduler(scheduler_type, optimizer, num_epochs):
    scheduler = None

    if scheduler_type == optim.lr_scheduler.ReduceLROnPlateau:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True, min_lr=5e-10
        )
    elif scheduler_type == optim.lr_scheduler.StepLR:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    elif scheduler_type == optim.lr_scheduler.CosineAnnealingLR:
        for group in optimizer.param_groups:
            group["initial_lr"] = lr
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs * 2, eta_min=5e-10
        )

    elif scheduler_type == optim.lr_scheduler.MultiStepLR:
        gamma = 0.1
        milestones = [150, 500, 700]
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )

    elif scheduler_type == optim.lr_scheduler.CyclicLR:
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=1e-10,  # Lower bound of learning rate
            max_lr=0.001,  # Upper bound of learning rate
            step_size_up=70,  # Number of training iterations in the increasing half of a cycle
            mode="triangular2",  # 'triangular' CLR policy
            cycle_momentum=True,  # Whether to cyclically vary momentum
            base_momentum=0.8,  # Lower bound for momentum
            max_momentum=0.9,  # Upper bound for momentum
        )

    return scheduler


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    data_sub_cfg,
    num_epochs=600,
    nb_epoch_log=10,
    scheduler_type: optim.lr_scheduler.LRScheduler = None,
    optimizer_type: optim.Optimizer = optim.Adam,
    criterion_type: nn.Module = nn.MSELoss,
    folder_prefix=None,
    use_pinn=None,
    pinn_lambda=0.2,
):
    """
    Train the given model using the provided training and validation data loaders.
    This function trains the model using Mean Squared Error (MSE) loss and the Adam optimizer.
    It also supports various learning rate schedulers. The training and validation losses and
    Mean Absolute Errors (MAE) are logged and plotted at the end of training.

    When use_pinn=True, a physics-informed loss term is added based on the Response Matrix
    that enforces physical consistency between predicted quadrupole offsets and BPM readings.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        device (torch.device): The device (CPU or GPU) to run the training on.
        num_epochs (int, optional): Number of epochs to train the model. Default is 600.
        nb_epoch_log (int, optional): Frequency of logging the training progress. Default is 10.
        scheduler_type (torch.optim.lr_scheduler.LRScheduler, optional): Learning rate scheduler. Default is CyclicLR.
        optimizer_type (torch.optim.Optimizer, optional): Optimizer to use for training. Default is Adam.
        criterion_type (torch.nn.Module, optional): Loss function to use for training. Default is MSELoss.
        folder_prefix (str, optional): Prefix for the training directory name.
        use_pinn (bool, optional): Enable physics-informed loss. If None, reads from config.
        pinn_lambda (float, optional): Weight for physics loss term. Default is 0.2.

    Returns:
        dict: A dictionary containing the training and validation losses and Mean Absolute Errors (MAE).
            - 'train_losses': List of training losses for each epoch.
            - 'val_losses': List of validation losses for each epoch.
            - 'train_maes': List of training MAEs for each epoch.
            - 'val_maes': List of validation MAEs for each epoch.
            - 'train_data_losses': List of data (MSE) losses for each epoch (if use_pinn=True).
            - 'train_phys_losses': List of physics losses for each epoch (if use_pinn=True).
            - 'val_data_losses': List of validation data losses for each epoch (if use_pinn=True).
            - 'val_phys_losses': List of validation physics losses for each epoch (if use_pinn=True).
    """

    merged_config = data_sub_cfg.get("merged_config", {})
    if use_pinn is None:
        use_pinn = merged_config.get("use_pinn", False)
    if pinn_lambda is None:
        pinn_lambda = merged_config.get("pinn_lambda", 0.2)

    if type(model) == SimpleFullyConnectedNetwork:
        lr = 0.0001
    elif type(model) == QuadErrorCorrectionLSTM:
        lr = 0.001
    elif type(model) == SimpleCNN:
        lr = 0.001

    # -------------------------------
    # Define loss function
    # -------------------------------
    if criterion_type == nn.MSELoss:
        criterion = nn.MSELoss(reduction="mean")
    elif criterion_type == nn.L1Loss:
        criterion = nn.L1Loss(reduction="mean")

    # -------------------------------
    # Define optimizer
    # -------------------------------
    if optimizer_type == optim.Adam:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    elif optimizer_type == optim.AdamW:
        optimizer = optim.AdamW(model.parameters(), lr=lr)

    # -------------------------------
    # Define learning rate scheduler
    # -------------------------------
    if scheduler_type is None:
        if type(model) == SimpleFullyConnectedNetwork:
            scheduler_type = optim.lr_scheduler.CosineAnnealingLR
        else:
            scheduler_type = optim.lr_scheduler.CyclicLR
            
    scheduler = get_scheduler(scheduler_type, optimizer, num_epochs)

    # Create a directory for the training session with the current date and time
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_prefix = f"{folder_prefix}_train" if folder_prefix else "train"
    save_dir = os.path.join(SAVE_DIR_TRAINING, f"{folder_prefix}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        scheduler_params = {
            "mode": scheduler.mode,
            "factor": scheduler.factor,
            "patience": scheduler.patience,
            "verbose": scheduler.verbose,
            "min_lr": scheduler.min_lrs,
        }
    elif isinstance(scheduler, optim.lr_scheduler.StepLR):
        scheduler_params = {"step_size": scheduler.step_size, "gamma": scheduler.gamma}
    elif isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR):
        scheduler_params = {"T_max": scheduler.T_max, "eta_min": scheduler.eta_min}
    elif isinstance(scheduler, optim.lr_scheduler.MultiStepLR):
        scheduler_params = {
            "milestones": scheduler.milestones,
            "gamma": scheduler.gamma,
        }
    elif isinstance(scheduler, optim.lr_scheduler.CyclicLR):
        scheduler_params = {
            "base_lr": scheduler.base_lrs[0],
            "max_lr": scheduler.max_lrs[0],
            "mode": scheduler.mode,
            "cycle_momentum": scheduler.cycle_momentum,
            "base_momentum": scheduler.base_momentums[0]
            if scheduler.cycle_momentum
            else None,
            "max_momentum": scheduler.max_momentums[0]
            if scheduler.cycle_momentum
            else None,
        }
    else:
        scheduler_params = {}

    config = {
        "data_shapes": model.data_shapes,
        "data_sub_cfg": data_sub_cfg,
        "data_sizes": {
            "train_batch_size": train_loader.batch_size,
            "validation_batch_size": val_loader.batch_size,
            "train_dataset_size": len(train_loader.dataset),
            "validation_dataset_size": len(val_loader.dataset),
        },
        "network_topology": {
            "input_shape": (-1,) + model.data_shapes["inputs_shape"][1:],
            "output_shape": (-1,) + model.data_shapes["targets_shape"][1:],
        },
        "training_parameters": {
            "num_epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "optimizer": optimizer_type.__name__,
            "learning_rate": lr,
            "criterion": criterion_type.__name__,
            "scheduler": scheduler_type.__name__ if scheduler else None,
        },
        "device": str(device),
        "random_seed": torch.initial_seed(),
        "timestamp": timestamp,
        "model_type": type(model).__name__,
        "model_parameters": {
            "input_size": getattr(
                model,
                "input_size",
                getattr(
                    getattr(model, "lstm", None),
                    "input_size",
                    getattr(model, "input_channels", None),
                ),
            ),
            "hidden_size": getattr(model, "hidden_size", None),
            "num_layers": getattr(model, "num_layers", None),
            "output_size": getattr(model, "output_size", None),
            "is_bidirectional": getattr(model, "is_bidirectional", None),
        },
        "optimizer_type": optimizer_type.__name__,
        "optimizer_params": optimizer.defaults
        if hasattr(optimizer, "defaults")
        else {},
        "scheduler_type": scheduler_type.__name__ if scheduler else None,
        "scheduler_params": scheduler_params,
        "criterion_type": criterion_type.__name__,
        "training_hyperparameters": {
            "num_epochs": num_epochs,
            "nb_epoch_log": nb_epoch_log,
        },
    }
    
    input_scaler = deserialize_minmax_scaler(data_sub_cfg["input_scaler_config"])
    target_scaler = deserialize_minmax_scaler(data_sub_cfg["target_scaler_config"])

    # print("Instantiating model with the following configuration:")
    # for key, value in config.items():
    #     print(f"  {key}: {value}")

    if use_pinn:
        print(f"merged_config : {merged_config}")
        print("\n[PINN] Generating physics-informed Response Matrix from Simulator...")
        temp_sim = SynchrotronSimulator(
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
        temp_sim.build_lattice()
        quad_cfg = merged_config.get("quad_errors", [])
        R_matrix = temp_sim.compute_response_matrix_y(quad_cfg, device=device)

        config["pinn"] = {
            "use_pinn": True,
            "pinn_lambda": pinn_lambda,
            "R_matrix_shape": list(R_matrix.shape),
            "R_matrix_device": str(R_matrix.device),
        }
        torch.save(R_matrix, os.path.join(save_dir, "R_matrix.pt"))

    # save config in json format
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # save config in yaml format
    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # -------------------------------
    # Training loop with evaluation
    # -------------------------------
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    train_data_losses = [] if use_pinn else None
    train_phys_losses = [] if use_pinn else None
    val_data_losses = [] if use_pinn else None
    val_phys_losses = [] if use_pinn else None
    best_val_loss = float("inf")

    if use_pinn:
        n_BPMs  = merged_config.get("n_FODO", model.data_shapes["raw_input_tensors_shape"][2])
        n_planes = model.data_shapes["raw_input_tensors_shape"][3]

    # sklearn inverse: X_original = (X_scaled - min_) / scale_
    # Target scaler: one entry per quad error output
    target_scale = torch.tensor(target_scaler.scale_, dtype=torch.float32, device=device)
    target_min   = torch.tensor(target_scaler.min_,   dtype=torch.float32, device=device)

    # Input scaler: fit on (N, n_planes=2), so index 1 = y-plane, shared across all BPMs
    input_scale_y = torch.tensor([input_scaler.scale_[1]], dtype=torch.float32, device=device)
    input_min_y   = torch.tensor([input_scaler.min_[1]],   dtype=torch.float32, device=device)
    
    print("Scaling factors loaded:")
    print(f"  - Target scale: {target_scale}")
    print(f"  - Target min: {target_min}")
    print(f"  - Input scale (y-plane): {input_scale_y}")
    print(f"  - Input min (y-plane): {input_min_y}")

    if use_pinn:
        # Precompute dataset-level mean BPM as approximation of nominal (no-error) orbit.
        # Valid because errors are zero-mean across the dataset, so E[bpm] = bpm_nominal.
        # This gives us the correct target: R @ errors = bpm_with_error - bpm_nominal
        bpm_nominal_sum = torch.zeros(n_BPMs, device=device)
        n_total = 0
        with torch.no_grad():
            for batch_inputs, _ in train_loader:
                batch_inputs = batch_inputs.to(device)
                inputs_4d = batch_inputs.view(batch_inputs.size(0), batch_inputs.size(1), n_BPMs, n_planes)
                y_scaled = inputs_4d[:, :, :, 1]          # (B, T, 8)
                bpm_nominal_sum += y_scaled.mean(dim=1).sum(dim=0)   # accumulate per-BPM mean
                n_total += batch_inputs.size(0)

        # --- GROUND TRUTH PHYSICS DIAGNOSTIC ---
        print("\n--- Running Ground Truth Physics Diagnostic ---")
        with torch.no_grad():
            for batch_inputs, batch_targets in train_loader:
                batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
                
                # 1. Unscale PERFECT True Targets
                true_phys = (batch_targets - target_min) / target_scale
                
                # 2. Project using R matrix
                true_bpm_est_phys = torch.matmul(true_phys, R_matrix.t())
                true_bpm_est_scaled = true_bpm_est_phys * input_scale_y
                
                # 3. Unscale True Inputs
                inputs_4d = batch_inputs.view(batch_inputs.size(0), batch_inputs.size(1), n_BPMs, n_planes)
                x_actual = inputs_4d[:, :, :, 1].mean(dim=1)
                x_actual_deviation = x_actual - input_min_y
                
                # 4. Check the Error
                gt_phys_loss = criterion(true_bpm_est_scaled, x_actual_deviation).item()
                print(f"[DIAGNOSTIC] Physics Loss of PERFECT Ground Truth: {gt_phys_loss:.6f}")
                break # Only need to check one batch
        print("----------------------------------------------\n")


    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        train_data_loss = 0.0
        train_phys_loss = 0.0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            outputs = model(batch_inputs)

            L_data = criterion(outputs, batch_targets)

            if use_pinn:
                # 1. Unscale predicted offsets to physical units (stay in torch for gradients)
                # sklearn inverse: X_original = (X_scaled - min_) / scale_
                outputs_physical = (outputs - target_min) / target_scale              # (B, n_errors), meters

                # 2. Project using R_matrix to get estimated BPM readings in physical units
                x_bpm_estimated_physical = torch.matmul(outputs_physical, R_matrix.t())  # (B, n_BPMs), meters

                # 3. Scale the estimated deviation to match L_data units
                # Deviations use only scale_, not min_ (constant offset cancels between bpm_with_error and bpm_nominal)
                x_bpm_estimated_scaled = x_bpm_estimated_physical * input_scale_y  # (B, n_BPMs)

                # 4. Actual vertical BPM readings are already scaled in [-1,1]
                inputs_4d = batch_inputs.view(batch_inputs.size(0), batch_inputs.size(1), n_BPMs, n_planes)
                y_scaled  = inputs_4d[:, :, :, 1]          # (B, T, 8) vertical plane, already in [-1,1]
                x_actual  = y_scaled.mean(dim=1)            # (B, 8) average over turns

                # 5. Physics loss: R @ errors = bpm_with_error - bpm_nominal
                # x_bpm_estimated_scaled is already the error-driven deviation (no nominal term)
                # Subtract precomputed nominal from actual readings to isolate error contribution
                # x_actual_deviation = x_actual - bpm_nominal_scaled      # (B, n_BPMs)
                
                # The physical nominal orbit is 0.0. 
                # In scaled space, 0.0 is exactly equal to input_min_y.
                x_actual_deviation = x_actual - input_min_y


                L_phys = criterion(x_bpm_estimated_scaled, x_actual_deviation)
                loss = L_data + pinn_lambda * L_phys
                
                # Update Accumulators for logging
                train_data_loss += L_data.item() * batch_inputs.size(0)
                train_phys_loss += L_phys.item() * batch_inputs.size(0)
            else:
                loss = L_data
                train_data_loss += L_data.item() * batch_inputs.size(0)
                train_phys_loss = 0.0

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_inputs.size(0)
            train_mae += compute_mae(outputs, batch_targets).item() * batch_inputs.size(
                0
            )

        train_loss /= len(train_loader.dataset)
        train_mae /= len(train_loader.dataset)
        if use_pinn:
            train_data_loss /= len(train_loader.dataset)
            train_phys_loss /= len(train_loader.dataset)

        # Evaluation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_data_loss = 0.0
        val_phys_loss = 0.0
        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)

                outputs = model(batch_inputs)
                L_data = criterion(outputs, batch_targets)

                if use_pinn:
                    # 1. Unscale predicted offsets to physical units
                    outputs_physical = (outputs - target_min) / target_scale              # (B, n_errors), meters

                    # 2. Project using R_matrix to get estimated BPM readings in physical units
                    x_bpm_estimated_physical = torch.matmul(outputs_physical, R_matrix.t())  # (B, n_BPMs), meters

                    # 3. Scale the estimated deviation — no min_ for deviations
                    x_bpm_estimated_scaled = x_bpm_estimated_physical * input_scale_y  # (B, n_BPMs)

                    # 4. Actual vertical BPM readings are already scaled in [-1,1]
                    inputs_4d = batch_inputs.view(batch_inputs.size(0), batch_inputs.size(1), n_BPMs, n_planes)
                    y_scaled  = inputs_4d[:, :, :, 1]          # (B, T, 8) vertical plane, already in [-1,1]
                    x_actual  = y_scaled.mean(dim=1)            # (B, 8) average over turns

                    # 5. Physics loss: same nominal subtraction as training
                    # x_actual_deviation = x_actual - bpm_nominal_scaled  # (B, n_BPMs)
                    # The physical nominal orbit is 0.0. 
                    # In scaled space, 0.0 is exactly equal to input_min_y.
                    x_actual_deviation = x_actual - input_min_y

                    L_phys = criterion(x_bpm_estimated_scaled, x_actual_deviation)
                    loss = L_data + pinn_lambda * L_phys
                    val_phys_loss += L_phys.item() * batch_inputs.size(0)
                    val_data_loss += L_data.item() * batch_inputs.size(0)
                    
                else:
                    loss = L_data
                    val_data_loss = 0.0
                    val_phys_loss = 0.0

                val_loss += loss.item() * batch_inputs.size(0)
                val_mae += compute_mae(
                    outputs, batch_targets
                ).item() * batch_inputs.size(0)

        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)
        if use_pinn:
            val_data_loss /= len(val_loader.dataset)
            val_phys_loss /= len(val_loader.dataset)

        if scheduler:
            # Step the scheduler
            if scheduler_type == optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Record losses for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_maes.append(train_mae)
        val_maes.append(val_mae)
        if use_pinn:
            train_data_losses.append(train_data_loss)
            train_phys_losses.append(train_phys_loss)
            val_data_losses.append(val_data_loss)
            val_phys_losses.append(val_phys_loss)

        # Print losses and MAE every nb_epoch_log epochs
        if (epoch + 1) % nb_epoch_log == 0:
            msg_parts = [
                f"Epoch [{epoch + 1}/{num_epochs}]",
            ]
            if use_pinn:
                msg_parts.append(
                    f"Train Loss: {train_loss:.6} (L_data={train_data_loss:.6}, L_phys={train_phys_loss:.6})"
                )
                msg_parts.append(
                    f"Val Loss: {val_loss:.6} (L_data={val_data_loss:.6}, L_phys={val_phys_loss:.6})"
                )
            else:
                msg_parts.append(
                    f"Train Loss: {train_loss:.6}, Val Loss: {val_loss:.6}"
                )

            msg_parts.append(f"Train MAE: {train_mae:.6}, Val MAE: {val_mae:.6}")

            if scheduler:
                msg_parts.append(f"lr: {scheduler.get_last_lr()}")

            print(", ".join(msg_parts))

        # Inside the epoch loop, after validation
        torch.save(model.state_dict(), os.path.join(save_dir, "latest.pt"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))

    # Training results
    train_results = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_maes": train_maes,
        "val_maes": val_maes,
    }
    if use_pinn:
        train_results["train_data_losses"] = train_data_losses
        train_results["train_phys_losses"] = train_phys_losses
        train_results["val_data_losses"] = val_data_losses
        train_results["val_phys_losses"] = val_phys_losses

    plot_training_results(num_epochs, train_losses, val_losses, train_maes, val_maes)

    if use_pinn:
        plot_pinn_losses(
            num_epochs,
            train_data_losses,
            train_phys_losses,
            val_data_losses,
            val_phys_losses,
            save_dir=save_dir,
        )

    return train_results
