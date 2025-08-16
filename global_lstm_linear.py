import gzip
import os
import pickle

import numpy as np
import torch
from torch import nn


class GlobalLSTMLinear:
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

        # ---------------- Transfer layer parameters ----------------
        # Learning rate for the optimizer
        self.optimizer_learning_rate = model_config["global_lstm_linear"][
            "optimizer_learning_rate"
        ]
        # Whether to use AMSGrad variant of Adam optimizer
        self.amsgrad = model_config["global_lstm_linear"]["amsgrad"]
        # Whether to export the model weights after training
        self.export_weights = model_config["global_lstm_linear"]["export_model"]
        # Dimension of the input to the transfer LSTM layer
        self.in_transfer_dim = model_config["global_lstm_linear"]["in_transfer_dim"]
        # Dimension of the output from the transfer LSTM layer
        self.out_transfer_dim = model_config["global_lstm_linear"]["out_transfer_dim"]
        # Number of LSTM layers in the global transfer LSTM
        self.transfer_layers = model_config["global_lstm_linear"]["n_layers"]
        # Dropout rate for LSTM layers
        self.dropout = model_config["global_lstm_linear"]["drop_rate"]

        # ---------------- Model dictionaries and task lists ----------------
        # List of main tasks (outer keys)
        self.multy_task_learning_list = list(self.Xtrain_tasks.keys())
        # Dictionary to hold sub-tasks for each main task
        self.sub_multy_task_learning_list = {}
        # Dictionaries to hold models, optimizers, activation layers, and loss histories for each (task, sub-task)
        (
            self.model_in_dict,  # Input Linear layers for each (task, sub_task)
            self.model_out_dict,  # Output Linear layers for each (task, sub_task)
            self.opt_dict,  # Optimizers for each (task, sub_task)
            self.signal_layer,  # Activation layers for each (task, sub_task)
            self.losses,  # Loss history for each (task, sub_task)
        ) = {}, {}, {}, {}, {}

        # ---------------- Global transfer LSTM layer ----------------
        # This LSTM is shared across all tasks/sub-tasks and maps from the input transfer dimension to the output transfer dimension
        self.global_transfer_lstm = (
            nn.LSTM(
                self.in_transfer_dim,
                self.out_transfer_dim,
                self.transfer_layers,
                batch_first=True,
                dropout=self.dropout,
            )
            .double()
            .to(self.device)
        )

        # Iterate over each main task
        for task in self.multy_task_learning_list:
            # Pre-allocate dictionaries for each main task
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

                # ------------- Linear layers for each (task, sub-task) -------------
                # Input Linear: maps input features to the transfer dimension
                self.model_in_dict[task][sub_task] = (
                    nn.Linear(n_in, self.in_transfer_dim).double().to(self.device)
                )
                # Output Linear: maps from transfer dimension to final output dimension
                self.model_out_dict[task][sub_task] = (
                    nn.Linear(self.out_transfer_dim, n_out).double().to(self.device)
                )
                # Activation layer (Tanh) for the output
                self.signal_layer[task][sub_task] = nn.Tanh().to(self.device)

                # ------------- Optimizer for each (task, sub-task) -------------
                # The optimizer updates the input linear, output linear, global transfer LSTM, and activation layer parameters
                self.opt_dict[task][sub_task] = torch.optim.Adam(
                    list(self.model_in_dict[task][sub_task].parameters())
                    + list(self.model_out_dict[task][sub_task].parameters())
                    + list(self.global_transfer_lstm.parameters())
                    + list(self.signal_layer[task][sub_task].parameters()),
                    lr=self.optimizer_learning_rate,
                    amsgrad=self.amsgrad,
                )
                # Print model and optimizer details for debugging
                print(
                    task,
                    sub_task,
                    self.model_in_dict[task][sub_task],
                    self.model_out_dict[task][sub_task],
                    self.global_transfer_lstm,
                    self.signal_layer[task][sub_task],
                    self.opt_dict[task][sub_task],
                )

    def train(self):
        """
        Train the model for the specified number of steps.
        For each step, iterate over all tasks and sub-tasks, sample batches, perform forward and backward passes, and update parameters.
        """
        for i in range(self.train_steps):
            for task in self.multy_task_learning_list:
                for sub_task in self.sub_multy_task_learning_list[task]:
                    # ----------------- Batch Sampling -----------------
                    # Randomly select batch indices for the current sub-task
                    start_ids = np.random.permutation(
                        list(
                            range(
                                self.Xtrain_tasks[task][sub_task].size(0)
                                - self.seq_len
                                - 1
                            )
                        )
                    )[: self.batch_size]
                    # Stack sequences for the batch: shape (batch_size, seq_len+1, n_features)
                    XYbatch = torch.stack(
                        [
                            self.Xtrain_tasks[task][sub_task][i : i + self.seq_len + 1]
                            for i in start_ids
                        ],
                        dim=0,
                    )
                    # Ytrain: targets (one-step ahead), Xtrain: inputs
                    Ytrain = XYbatch[:, 1:, :]  # Targets: next time step
                    Xtrain = XYbatch[:, :-1, :]  # Inputs: current time step

                    # ----------------- Forward and Backward Pass -----------------
                    # Reset gradients before each batch
                    self.opt_dict[task][sub_task].zero_grad()

                    # Pass input through the input linear layer
                    in_pred = self.model_in_dict[task][sub_task](Xtrain)

                    # Initialize hidden and cell states for the global transfer LSTM
                    (hidden, cell) = self.get_hidden(
                        self.batch_size, self.transfer_layers, self.in_transfer_dim
                    )
                    # Pass through the global transfer LSTM
                    global_pred, _ = self.global_transfer_lstm(in_pred, (hidden, cell))

                    # Pass through the output linear layer and activation
                    preds = self.signal_layer[task][sub_task](
                        self.model_out_dict[task][sub_task](global_pred)
                    )

                    # Compute loss using the criterion (Sharpe ratio loss)
                    loss = self.criterion(preds, Ytrain)
                    # Store loss for monitoring
                    self.losses[task][sub_task].append(loss.item())

                    # Backpropagation and parameter update
                    loss.backward()
                    self.opt_dict[task][sub_task].step()

            # Print progress every 100 steps
            if (i % 100) == 1:
                print(i)

        # ----------------- Export Model Weights (if enabled) -----------------
        if self.export_weights:
            for task in self.multy_task_learning_list:
                for sub_task in self.sub_multy_task_learning_list[task]:
                    # Save input linear layer weights
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
                    # Save output linear layer weights
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
                # Save global transfer LSTM weights (once per task)
                torch.save(
                    self.global_transfer_lstm,
                    self.export_path
                    + task
                    + "_"
                    + self.export_label
                    + "_globaltransferlstm.pt",
                )

    def predict(self, x_test):
        """
        Generate predictions for the provided test data.
        Args:
            x_test (dict): Nested dictionary of test data, organized by task and sub_task.
        Returns:
            y_pred (dict): Nested dictionary of predictions, organized by task and sub_task.
        """
        y_pred = {}
        for task in self.multy_task_learning_list:
            y_pred[task] = {}
            for sub_task in self.sub_multy_task_learning_list[task]:
                # Reshape input to add a batch dimension (batch size = 1)
                xflat = x_test[task][sub_task].view(
                    1, -1, x_test[task][sub_task].size(1)
                )
                with torch.autograd.no_grad():
                    # Pass input through the input linear layer for this (task, sub_task)
                    # Exclude the last time step to align with training
                    in_pred = self.model_in_dict[task][sub_task](xflat[:, :-1])
                    # Initialize hidden and cell states for the global transfer LSTM
                    (hidden, cell) = self.get_hidden(
                        1, self.transfer_layers, self.in_transfer_dim
                    )
                    # Pass through the global transfer LSTM
                    global_pred, _ = self.global_transfer_lstm(in_pred, (hidden, cell))
                    # Pass through the output linear layer and activation for this (task, sub_task)
                    y_pred[task][sub_task] = self.signal_layer[task][sub_task](
                        self.model_out_dict[task][sub_task](global_pred)
                    )

        # Return the nested dictionary of predictions
        return y_pred

    def avg_sharpe_ratio(self, output, target):
        """
        Custom loss function based on the average Sharpe ratio.
        Penalizes volatility and transaction costs.
        Args:
            output (Tensor): Model predictions.
            target (Tensor): Ground truth targets.
        Returns:
            loss (Tensor): Scalar loss value (negative Sharpe ratio).
        """
        slip = 0.0005 * 0.00  # Slippage cost (set to zero here)
        bp = 0.0020 * 0.00  # Basis point cost (set to zero here)
        # Element-wise product of predictions and targets (returns)
        rets = torch.mul(output, target)
        # Transaction costs: difference between consecutive predictions
        tc = torch.abs(output[:, 1:, :] - output[:, :-1, :]) * (bp + slip)
        # Pad transaction costs to match sequence length
        tc = torch.cat(
            [
                torch.zeros(output.size(0), 1, output.size(2)).double().to(self.device),
                tc,
            ],
            dim=1,
        )
        # Subtract transaction costs from returns
        rets = rets - tc
        # Compute mean and standard deviation of returns
        avg_rets = torch.mean(rets)
        vol_rets = torch.std(rets)
        # Negative Sharpe ratio as loss
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
        return (
            torch.zeros(n_layers, batch_size, n_hi).double().to(self.device),
            torch.zeros(n_layers, batch_size, n_hi).double().to(self.device),
        )

    def save_model(self, folder_path):
        """
        Save the entire model state as a pickle file
        """
        # Ensure the folder exists
        os.makedirs(folder_path, exist_ok=True)

        # Generate filename with training steps
        filename = f"global_lstm_linear_trainsteps_{self.train_steps}.pkl.gz"
        filepath = os.path.join(folder_path, filename)

        model_state = {
            "model_in_dict": self.model_in_dict,
            "model_out_dict": self.model_out_dict,
            "global_transfer_lstm": self.global_transfer_lstm,
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
            "transfer_layers": self.transfer_layers,
            "dropout": self.dropout,
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
        instance.global_transfer_lstm = model_state["global_transfer_lstm"]
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
        instance.transfer_layers = model_state["transfer_layers"]
        instance.dropout = model_state["dropout"]

        # Set the criterion function
        instance.criterion = instance.avg_sharpe_ratio

        # Note: Xtrain_tasks is not saved as it's not needed for prediction
        # It was only used during training for data access

        return instance
