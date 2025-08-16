import gzip
import os
import pickle

import numpy as np
import torch
from torch import nn


class GlobalLinearLinear:
    def __init__(self, x_tasks, model_config):
        """
        Initialize the GlobalLinearLinear model.

        Args:
            x_tasks (dict): Nested dictionary of training data, organized by task and sub_task.
            model_config (dict): Configuration dictionary containing model and training parameters.

        This model implements a global transfer learning architecture using linear layers:
        - Each (task, sub_task) pair has its own input and output linear layers.
        - A global transfer linear layer is shared across all tasks/sub_tasks, mapping from a shared latent space.
        - Tanh activation is applied after the output linear layer.
        - Each (task, sub_task) has its own optimizer, which also updates the global transfer layer.
        """
        # General training parameters
        self.criterion = self.avg_sharpe_ratio  # Custom Sharpe ratio loss function
        self.Xtrain_tasks = x_tasks  # Training data for all tasks
        self.train_steps = model_config["train_steps"]  # Total training steps
        self.tasks_train_steps = model_config[
            "tasks_train_steps"
        ]  # Per-task steps (if used)
        self.batch_size = model_config["batch_size"]  # Batch size for training
        self.seq_len = model_config["seq_len"]  # Sequence length for each sample
        self.device = model_config["device"]  # Device to run the model on
        self.export_path = model_config["export_path"]  # Path to export model weights
        self.export_label = model_config["export_label"]  # Label for exported weights

        # Model-specific parameters for the global linear transfer model
        self.optimizer_learning_rate = model_config["global_linear_linear"][
            "optimizer_learning_rate"
        ]  # Learning rate for optimizer
        self.amsgrad = model_config["global_linear_linear"][
            "amsgrad"
        ]  # Use AMSGrad variant of Adam
        self.export_weights = model_config["global_linear_linear"][
            "export_weights"
        ]  # Whether to export weights
        self.in_transfer_dim = model_config["global_linear_linear"][
            "in_transfer_dim"
        ]  # Inner dimension for transfer matrix (input to global transfer)
        self.out_transfer_dim = model_config["global_linear_linear"][
            "out_transfer_dim"
        ]  # Inner dimension for transfer matrix (output from global transfer)

        # Prepare lists and dictionaries to manage tasks and sub-tasks
        self.multy_task_learning_list = list(
            self.Xtrain_tasks.keys()
        )  # List of main tasks
        self.sub_multy_task_learning_list = {}  # Dict to hold sub-tasks for each main task

        # Dictionaries to hold models, optimizers, activation layers, and loss histories for each (task, sub-task)
        (
            self.model_in_dict,  # Input linear layers for each (task, sub_task)
            self.model_out_dict,  # Output linear layers for each (task, sub_task)
            self.opt_dict,  # Optimizers for each (task, sub_task)
            self.signal_layer,  # Activation layers for each (task, sub_task)
            self.losses,  # Loss history for each (task, sub_task)
        ) = {}, {}, {}, {}, {}

        # Global transfer linear layer shared across all tasks/sub_tasks
        self.global_transfer_linear = (
            nn.Linear(self.in_transfer_dim, self.out_transfer_dim)
            .double()
            .to(self.device)
        )

        # Iterate over each main task
        for task in self.multy_task_learning_list:
            # Initialize dictionaries for each main task
            (
                self.model_in_dict[task],
                self.model_out_dict[task],
                self.signal_layer[task],
                self.opt_dict[task],
                self.losses[task],
            ) = {}, {}, {}, {}, {}

            # Get the list of sub-tasks for this main task
            self.sub_multy_task_learning_list[task] = list(
                self.Xtrain_tasks[task].keys()
            )

            # Iterate over each sub-task for the current main task
            for sub_task in self.sub_multy_task_learning_list[task]:
                # Initialize the loss history list for this (task, sub-task)
                self.losses[task][sub_task] = []

                # Determine the number of input and output features from the data shape
                n_in = self.Xtrain_tasks[task][sub_task].shape[
                    1
                ]  # Number of input features
                n_out = self.Xtrain_tasks[task][sub_task].shape[
                    1
                ]  # Number of output features (same as input here)

                # Create input linear layer for this (task, sub-task)
                self.model_in_dict[task][sub_task] = (
                    nn.Linear(n_in, self.in_transfer_dim).double().to(self.device)
                )
                # Create output linear layer for this (task, sub-task)
                self.model_out_dict[task][sub_task] = (
                    nn.Linear(self.out_transfer_dim, n_out).double().to(self.device)
                )
                # Add a Tanh activation layer for this (task, sub-task)
                self.signal_layer[task][sub_task] = nn.Tanh().to(self.device)

                # Create an Adam optimizer for the parameters of the input, output, global transfer, and activation layers
                self.opt_dict[task][sub_task] = torch.optim.Adam(
                    list(self.model_in_dict[task][sub_task].parameters())
                    + list(self.model_out_dict[task][sub_task].parameters())
                    + list(self.global_transfer_linear.parameters())
                    + list(self.signal_layer[task][sub_task].parameters()),
                    lr=self.optimizer_learning_rate,
                    amsgrad=self.amsgrad,
                )

                # Print out the configuration for this (task, sub-task) for debugging/verification
                print(
                    task,
                    sub_task,
                    self.model_in_dict[task][sub_task],
                    self.model_out_dict[task][sub_task],
                    self.global_transfer_linear,
                    self.signal_layer[task][sub_task],
                    self.opt_dict[task][sub_task],
                )

    def train(self):
        """
        Train the global linear-linear transfer model for all tasks and sub-tasks.

        This method iterates over the specified number of training steps. For each step, it loops through all tasks and their sub-tasks,
        samples random batches from the training data, performs a forward pass through the model (input linear layer, global transfer layer,
        output linear layer, and activation), computes the loss, and updates the model parameters using backpropagation and the optimizer.

        Optionally, after training, model weights are exported to disk if export_weights is True.
        """
        for i in range(self.train_steps):
            # Loop over each main task
            for task in self.multy_task_learning_list:
                # Loop over each sub-task for the current main task
                for sub_task in self.sub_multy_task_learning_list[task]:
                    # Randomly select batch start indices for the current sub-task
                    start_ids = np.random.permutation(
                        list(
                            range(
                                self.Xtrain_tasks[task][sub_task].size(0)
                                - self.seq_len
                                - 1
                            )
                        )
                    )[: self.batch_size]

                    # Stack sequences of length (seq_len + 1) for each batch index
                    XYbatch = torch.stack(
                        [
                            self.Xtrain_tasks[task][sub_task][i : i + self.seq_len + 1]
                            for i in start_ids
                        ],
                        dim=0,
                    )

                    # Ytrain: target values (one-step ahead for each sequence in the batch)
                    Ytrain = XYbatch[:, 1:, :]  # shape: (batch_size, seq_len, features)
                    # Xtrain: input values (all but the last time step in each sequence)
                    Xtrain = XYbatch[
                        :, :-1, :
                    ]  # shape: (batch_size, seq_len, features)

                    # Reset gradients for the optimizer before the backward pass
                    self.opt_dict[task][sub_task].zero_grad()

                    # Forward pass:
                    # 1. Pass input through the input linear layer
                    in_pred = self.model_in_dict[task][sub_task](Xtrain)
                    # 2. Pass through the global transfer linear layer (shared across all tasks)
                    global_pred = self.global_transfer_linear(in_pred)
                    # 3. Pass through the output linear layer and activation function
                    preds = self.signal_layer[task][sub_task](
                        self.model_out_dict[task][sub_task](global_pred)
                    )

                    # Compute loss using the specified criterion (e.g., average Sharpe ratio)
                    loss = self.criterion(preds, Ytrain)
                    # Store the loss value for monitoring/tracking
                    self.losses[task][sub_task].append(loss.item())

                    # Backward pass: compute gradients
                    loss.backward()
                    # Update model parameters using the optimizer
                    self.opt_dict[task][sub_task].step()

            # Print progress every 100 iterations (when i mod 100 == 1)
            if (i % 100) == 1:
                print(i)

        # Optionally export model weights after training if export_weights is True
        if self.export_weights:
            for task in self.multy_task_learning_list:
                for sub_task in self.sub_multy_task_learning_list[task]:
                    # Save the input linear layer for this (task, sub-task)
                    torch.save(
                        self.model_in_dict[task][sub_task],
                        self.export_path
                        + task
                        + "_"
                        + sub_task
                        + "_"
                        + self.export_label
                        + "_intransferlinear.pt",
                    )
                    # Save the output linear layer for this (task, sub-task)
                    torch.save(
                        self.model_out_dict[task][sub_task],
                        self.export_path
                        + task
                        + "_"
                        + sub_task
                        + "_"
                        + self.export_label
                        + "_outtransferlinear.pt",
                    )
                # Save the global transfer linear layer (shared across all sub-tasks of this task)
                torch.save(
                    self.global_transfer_linear,
                    self.export_path
                    + task
                    + "_"
                    + self.export_label
                    + "_globaltransferlinear.pt",
                )

    def predict(self, x_test):
        """
        Generate predictions for the given test data.

        Args:
            x_test (dict): Nested dictionary of test input tensors, organized by task and sub_task.

        Returns:
            y_pred (dict): Nested dictionary of predictions, organized by task and sub_task.
        """
        y_pred = {}  # Initialize dictionary to store predictions for each task and sub_task
        for task in self.multy_task_learning_list:
            y_pred[task] = {}  # Initialize sub-dictionary for each task
            for sub_task in self.sub_multy_task_learning_list[task]:
                # Reshape input to add a batch dimension (batch size = 1)
                # xflat shape: (1, sequence_length, feature_dim)
                xflat = x_test[task][sub_task].view(
                    1, -1, x_test[task][sub_task].size(1)
                )
                # Disable gradient computation for inference
                with torch.autograd.no_grad():
                    # Pass input through the input linear layer for this (task, sub_task)
                    # Exclude the last time step to align with training
                    in_pred = self.model_in_dict[task][sub_task](xflat[:, :-1])
                    # Pass through the global transfer linear layer (shared across sub-tasks)
                    global_pred = self.global_transfer_linear(in_pred)
                    # Pass through the output linear layer and activation for this (task, sub_task)
                    y_pred[task][sub_task] = self.signal_layer[task][sub_task](
                        self.model_out_dict[task][sub_task](global_pred)
                    )

        # Return the nested dictionary of predictions
        return y_pred

    def avg_sharpe_ratio(self, output, target):
        slip = 0.0005 * 0.00
        bp = 0.0020 * 0.00
        rets = torch.mul(output, target)
        tc = torch.abs(output[:, 1:, :] - output[:, :-1, :]) * (bp + slip)
        tc = torch.cat(
            [
                torch.zeros(output.size(0), 1, output.size(2)).double().to(self.device),
                tc,
            ],
            dim=1,
        )
        rets = rets - tc
        avg_rets = torch.mean(rets)
        vol_rets = torch.std(rets)
        loss = torch.neg(torch.div(avg_rets, vol_rets))
        return loss.mean()

    def save_model(self, folder_path):
        """
        Save the entire model state as a pickle file
        """
        # Ensure the folder exists
        os.makedirs(folder_path, exist_ok=True)

        # Generate filename with training steps
        filename = f"global_linear_linear_trainsteps_{self.train_steps}.pkl.gz"
        filepath = os.path.join(folder_path, filename)

        model_state = {
            "model_in_dict": self.model_in_dict,
            "model_out_dict": self.model_out_dict,
            "global_transfer_linear": self.global_transfer_linear,
            "signal_layer": self.signal_layer,
            "opt_dict": self.opt_dict,
            "losses": self.losses,
            "multy_task_learning_list": self.multy_task_learning_list,
            "sub_multy_task_learning_list": self.sub_multy_task_learning_list,
            "train_steps": self.train_steps,
            "tasks_train_steps": self.tasks_train_steps,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "device": self.device,
            "export_path": self.export_path,
            "export_label": self.export_label,
            "optimizer_learning_rate": self.optimizer_learning_rate,
            "amsgrad": self.amsgrad,
            "export_weights": self.export_weights,
            "in_transfer_dim": self.in_transfer_dim,
            "out_transfer_dim": self.out_transfer_dim,
        }

        with gzip.open(filepath, "wb") as f:
            pickle.dump(model_state, f)

        print(f"Model saved to: {filepath}")

    @classmethod
    def load_model(cls, filepath):
        """
        Load a saved model from pickle file
        """
        with gzip.open(filepath, "rb") as f:
            model_state = pickle.load(f)

        # Create a new instance
        instance = cls.__new__(cls)

        # Restore all the model state
        instance.model_in_dict = model_state["model_in_dict"]
        instance.model_out_dict = model_state["model_out_dict"]
        instance.global_transfer_linear = model_state["global_transfer_linear"]
        instance.signal_layer = model_state["signal_layer"]
        instance.opt_dict = model_state["opt_dict"]
        instance.losses = model_state["losses"]
        instance.multy_task_learning_list = model_state["multy_task_learning_list"]
        instance.sub_multy_task_learning_list = model_state[
            "sub_multy_task_learning_list"
        ]
        instance.train_steps = model_state["train_steps"]
        instance.tasks_train_steps = model_state["tasks_train_steps"]
        instance.batch_size = model_state["batch_size"]
        instance.seq_len = model_state["seq_len"]
        instance.device = model_state["device"]
        instance.export_path = model_state["export_path"]
        instance.export_label = model_state["export_label"]
        instance.optimizer_learning_rate = model_state["optimizer_learning_rate"]
        instance.amsgrad = model_state["amsgrad"]
        instance.export_weights = model_state["export_weights"]
        instance.in_transfer_dim = model_state["in_transfer_dim"]
        instance.out_transfer_dim = model_state["out_transfer_dim"]

        # Set the criterion function
        instance.criterion = instance.avg_sharpe_ratio

        # Note: Xtrain_tasks is not saved as it's not needed for prediction
        # It was only used during training for data access

        return instance
