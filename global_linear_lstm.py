import gzip
import os
import pickle

import numpy as np
import torch
from torch import nn


class GlobalLinearLSTM:
    def __init__(self, x_tasks, model_config):
        # ---------------- General parameters ----------------
        # Set the loss criterion to a custom Sharpe ratio loss function
        self.criterion = self.avg_sharpe_ratio
        # Store the training data for all tasks and sub-tasks
        self.Xtrain_tasks = x_tasks
        # Number of total training steps
        self.train_steps = model_config["train_steps"]
        # Number of training steps per task (if used)
        self.tasks_train_steps = model_config["tasks_train_steps"]
        # Batch size for training
        self.batch_size = model_config["batch_size"]
        # Sequence length for each training sample
        self.seq_len = model_config["seq_len"]
        # Device to run the model on (e.g., 'cuda' or 'cpu')
        self.device = model_config["device"]
        # Path to export model weights
        self.export_path = model_config["export_path"]
        # Label for exported weights
        self.export_label = model_config["export_label"]

        # ---------------- Model-specific parameters ----------------
        # Learning rate for the optimizer
        self.optimizer_learning_rate = model_config["global_linear_lstm"][
            "optimizer_learning_rate"
        ]
        # Whether to use AMSGrad variant of Adam optimizer
        self.amsgrad = model_config["global_linear_lstm"]["amsgrad"]
        # Whether to export the model
        self.export_model = model_config["global_linear_lstm"]["export_model"]
        # Number of layers in the input LSTM
        self.in_n_layers = model_config["global_linear_lstm"]["in_n_layers"]
        # Number of layers in the output LSTM
        self.out_n_layers = model_config["global_linear_lstm"]["out_n_layers"]
        # Hidden size for the output LSTM
        self.out_n_hi = model_config["global_linear_lstm"]["out_n_hi"]
        # Dropout rate for LSTM layers
        self.drop_rate = model_config["global_linear_lstm"]["drop_rate"]

        # ---------------- Transfer layer parameters ----------------
        # Dimension of the input to the transfer linear layer
        self.in_transfer_dim = model_config["global_linear_lstm"]["in_transfer_dim"]
        # Dimension of the output from the transfer linear layer
        self.out_transfer_dim = model_config["global_linear_lstm"]["out_transfer_dim"]

        # ---------------- Global transfer linear layer ----------------
        # This layer is shared across all tasks/sub-tasks and maps from the input transfer dimension to the output transfer dimension
        self.transfer_linear = (
            nn.Linear(self.in_transfer_dim, self.out_transfer_dim)
            .double()
            .to(self.device)
        )

        # ---------------- Model dictionaries and task lists ----------------
        # List of main tasks (outer keys)
        self.multy_task_learning_list = list(self.Xtrain_tasks.keys())
        # Dictionary to hold sub-tasks for each main task
        self.sub_multy_task_learning_list = {}
        # Dictionaries to hold models, optimizers, activation layers, and loss histories for each (task, sub-task)
        (
            self.model_in_dict,  # Input LSTM layers for each (task, sub_task)
            self.model_out_dict,  # Output LSTM layers for each (task, sub_task)
            self.model_lin_dict,  # Output Linear layers for each (task, sub_task)
            self.opt_dict,  # Optimizers for each (task, sub_task)
            self.signal_layer,  # Activation layers for each (task, sub_task)
            self.losses,  # Loss history for each (task, sub_task)
        ) = {}, {}, {}, {}, {}, {}

        # Iterate over each main task
        for task in self.multy_task_learning_list:
            # Pre-allocate dictionaries for each main task
            (
                self.model_in_dict[task],
                self.model_out_dict[task],
                self.model_lin_dict[task],
                self.signal_layer[task],
                self.opt_dict[task],
                self.losses[task],
            ) = {}, {}, {}, {}, {}, {}
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

                # ------------- LSTM + Linear layers for each (task, sub-task) -------------
                # Unpack LSTM configuration for clarity
                in_n_layers, out_n_layers, out_n_hi = (
                    self.in_n_layers,
                    self.out_n_layers,
                    self.out_n_hi,
                )
                # Input LSTM: maps input features to the transfer dimension
                self.model_in_dict[task][sub_task] = (
                    nn.LSTM(
                        n_in,
                        self.in_transfer_dim,
                        in_n_layers,
                        batch_first=True,
                        dropout=self.drop_rate,
                    )
                    .double()
                    .to(self.device)
                )
                # Output LSTM: maps from transfer dimension to hidden output
                self.model_out_dict[task][sub_task] = (
                    nn.LSTM(
                        self.out_transfer_dim,
                        out_n_hi,
                        out_n_layers,
                        batch_first=True,
                        dropout=self.drop_rate,
                    )
                    .double()
                    .to(self.device)
                )
                # Output Linear: maps from hidden output to final output dimension
                self.model_lin_dict[task][sub_task] = (
                    nn.Linear(out_n_hi, n_out).double().to(self.device)
                )
                # Activation layer (Tanh) for the final output
                self.signal_layer[task][sub_task] = nn.Tanh().to(self.device)

                # ------------- Optimizer for each (task, sub-task) -------------
                # The optimizer updates all parameters for the input LSTM, output LSTM, output linear, transfer linear, and activation layer
                self.opt_dict[task][sub_task] = torch.optim.Adam(
                    list(self.model_in_dict[task][sub_task].parameters())
                    + list(self.model_out_dict[task][sub_task].parameters())
                    + list(self.model_lin_dict[task][sub_task].parameters())
                    + list(self.transfer_linear.parameters())
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
                    self.model_lin_dict[task][sub_task],
                    self.transfer_linear,
                    self.signal_layer[task][sub_task],
                    self.opt_dict[task][sub_task],
                )

    def train(self):
        # Main training loop over the total number of training steps
        for i in range(self.train_steps):
            # Loop over each task
            for task in self.multy_task_learning_list:
                # Loop over each sub-task for the current task
                for sub_task in self.sub_multy_task_learning_list[task]:
                    # -------------------- Batch Sampling --------------------
                    # Randomly select batch indices for the current (task, sub_task)
                    start_ids = np.random.permutation(
                        list(
                            range(
                                self.Xtrain_tasks[task][sub_task].size(0)
                                - self.seq_len
                                - 1
                            )
                        )
                    )[: self.batch_size]
                    # Stack the selected sequences into a batch tensor
                    # Each batch is a sequence of length (seq_len + 1)
                    XYbatch = torch.stack(
                        [
                            self.Xtrain_tasks[task][sub_task][i : i + self.seq_len + 1]
                            for i in start_ids
                        ],
                        dim=0,
                    )
                    # Ytrain: targets for prediction (one-step ahead for each sequence)
                    Ytrain = XYbatch[:, 1:, :]  # shape: (batch_size, seq_len, features)
                    # Xtrain: input sequences (excluding the last step)
                    Xtrain = XYbatch[
                        :, :-1, :
                    ]  # shape: (batch_size, seq_len, features)

                    # -------------------- Forward Pass --------------------
                    # Reset gradients for the optimizer before the forward pass
                    self.opt_dict[task][sub_task].zero_grad()

                    # Initialize hidden and cell states for the input LSTM
                    (hidden, cell) = self.get_hidden(
                        self.batch_size, self.in_n_layers, self.in_transfer_dim
                    )
                    # Pass input through the input LSTM for this (task, sub_task)
                    in_pred, _ = self.model_in_dict[task][sub_task](
                        Xtrain, (hidden, cell)
                    )

                    # Pass the LSTM output through the global transfer linear layer
                    global_pred = self.transfer_linear(in_pred)

                    # Initialize hidden and cell states for the output LSTM
                    (hidden, cell) = self.get_hidden(
                        self.batch_size, self.out_n_layers, self.out_n_hi
                    )
                    # Pass through the output LSTM for this (task, sub_task)
                    hidden_pred, _ = self.model_out_dict[task][sub_task](
                        global_pred, (hidden, cell)
                    )

                    # Pass through the output linear layer and activation (Tanh)
                    preds = self.signal_layer[task][sub_task](
                        self.model_lin_dict[task][sub_task](hidden_pred)
                    )

                    # -------------------- Loss and Optimization --------------------
                    # Compute the loss using the custom criterion (e.g., Sharpe ratio loss)
                    loss = self.criterion(preds, Ytrain)
                    # Store the loss value for monitoring
                    self.losses[task][sub_task].append(loss.item())

                    # Backpropagate the loss to compute gradients
                    loss.backward()
                    # Update model parameters using the optimizer
                    self.opt_dict[task][sub_task].step()

            # Print progress every 100 steps for monitoring
            if (i % 100) == 1:
                print(i)

        # -------------------- Model Export --------------------
        # If export_model is enabled, save model weights for each (task, sub_task)
        if self.export_model:
            for task in self.multy_task_learning_list:
                for sub_task in self.sub_multy_task_learning_list[task]:
                    # Save the input LSTM weights
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
                    # Save the output LSTM weights
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
                    # Save the output linear layer weights
                    torch.save(
                        self.model_lin_dict[task][sub_task],
                        self.export_path
                        + task
                        + "_"
                        + sub_task
                        + "_"
                        + self.export_label
                        + "_outtransferlinear.pt",
                    )
                # Save the global transfer linear layer weights (shared across sub-tasks)
                torch.save(
                    self.transfer_linear,
                    self.export_path
                    + task
                    + "_"
                    + self.export_label
                    + "_transferlinear.pt",
                )

    def predict(self, x_test):
        """
        Generate predictions for the given test data using the trained model.

        Args:
            x_test (dict): Nested dictionary of test inputs, organized by task and sub_task.
                           Each entry should be a tensor of shape (sequence_length, feature_dim).

        Returns:
            y_pred (dict): Nested dictionary of predictions, organized by task and sub_task.
        """
        y_pred = {}  # Initialize dictionary to store predictions for each task and sub_task

        # Loop over each main task
        for task in self.multy_task_learning_list:
            y_pred[task] = {}  # Initialize sub-dictionary for each task

            # Loop over each sub-task for the current main task
            for sub_task in self.sub_multy_task_learning_list[task]:
                # Reshape input to add a batch dimension (batch size = 1)
                # xflat shape: (1, sequence_length, feature_dim)
                xflat = x_test[task][sub_task].view(
                    1, -1, x_test[task][sub_task].size(1)
                )

                # Disable gradient computation for inference
                with torch.autograd.no_grad():
                    # Initialize hidden and cell states for the input LSTM
                    (hidden, cell) = self.get_hidden(
                        1, self.in_n_layers, self.in_transfer_dim
                    )

                    # Pass input through the input LSTM for this (task, sub_task)
                    # Exclude the last time step to align with training
                    in_pred, _ = self.model_in_dict[task][sub_task](
                        xflat[:, :-1], (hidden, cell)
                    )

                    # Pass through the global transfer linear layer (shared across sub-tasks)
                    global_pred = self.transfer_linear(in_pred)

                    # Initialize hidden and cell states for the output LSTM
                    (hidden, cell) = self.get_hidden(
                        1, self.out_n_layers, self.out_n_hi
                    )

                    # Pass through the output LSTM for this (task, sub_task)
                    hidden_pred, _ = self.model_out_dict[task][sub_task](
                        global_pred, (hidden, cell)
                    )

                    # Pass through the output linear layer and activation for this (task, sub_task)
                    y_pred[task][sub_task] = self.signal_layer[task][sub_task](
                        self.model_lin_dict[task][sub_task](hidden_pred)
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

    def get_hidden(self, batch_size, n_layers, n_hi):
        """
        Initialize the hidden and cell states for an LSTM.

        Args:
            batch_size (int): The batch size for the input.
            n_layers (int): Number of LSTM layers.
            n_hi (int): Hidden size (number of features in the hidden state).

        Returns:
            tuple: A tuple containing two tensors:
                - hidden state tensor of shape (n_layers, batch_size, n_hi)
                - cell state tensor of shape (n_layers, batch_size, n_hi)
            Both tensors are initialized to zeros, use double precision, and are moved to the correct device.
        """
        # Create a tensor of zeros for the hidden state, with the correct shape and device
        hidden = torch.zeros(n_layers, batch_size, n_hi).double().to(self.device)
        # Create a tensor of zeros for the cell state, with the correct shape and device
        cell = torch.zeros(n_layers, batch_size, n_hi).double().to(self.device)
        return (hidden, cell)

    def save_model(self, folder_path):
        """
        Save the entire model state as a pickle file
        """
        # Ensure the folder exists
        os.makedirs(folder_path, exist_ok=True)

        # Generate filename with training steps
        filename = f"global_linear_lstm_trainsteps_{self.train_steps}.pkl.gz"
        filepath = os.path.join(folder_path, filename)

        model_state = {
            "model_in_dict": self.model_in_dict,
            "model_out_dict": self.model_out_dict,
            "model_lin_dict": self.model_lin_dict,
            "transfer_linear": self.transfer_linear,
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
            "export_model": self.export_model,
            "in_n_layers": self.in_n_layers,
            "out_n_layers": self.out_n_layers,
            "out_n_hi": self.out_n_hi,
            "drop_rate": self.drop_rate,
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
        instance.model_lin_dict = model_state["model_lin_dict"]
        instance.transfer_linear = model_state["transfer_linear"]
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
        instance.export_model = model_state["export_model"]
        instance.in_n_layers = model_state["in_n_layers"]
        instance.out_n_layers = model_state["out_n_layers"]
        instance.out_n_hi = model_state["out_n_hi"]
        instance.drop_rate = model_state["drop_rate"]
        instance.in_transfer_dim = model_state["in_transfer_dim"]
        instance.out_transfer_dim = model_state["out_transfer_dim"]

        # Set the criterion function
        instance.criterion = instance.avg_sharpe_ratio

        # Note: Xtrain_tasks is not saved as it's not needed for prediction
        # It was only used during training for data access

        return instance
