import gzip
import os
import pickle

import numpy as np
import torch
from torch import nn


class NoTransferLSTM:
    def __init__(self, x_tasks, model_config):
        # Set the loss criterion (custom Sharpe ratio function)
        self.criterion = self.avg_sharpe_ratio  # Could use nn.MSELoss().cuda() for MSE
        # Store the training data for all tasks
        self.Xtrain_tasks = x_tasks
        # Number of training steps for the overall model
        self.train_steps = model_config["train_steps"]
        # Number of training steps for each task
        self.tasks_train_steps = model_config["tasks_train_steps"]
        # Batch size for training
        self.batch_size = model_config["batch_size"]
        # Sequence length for LSTM input
        self.seq_len = model_config["seq_len"]
        # Device to run the model on (e.g., 'cuda' or 'cpu')
        self.device = model_config["device"]
        # Path to export the trained model
        self.export_path = model_config["export_path"]
        # Label for exported model files
        self.export_label = model_config["export_label"]

        # Model-specific parameters from the config
        self.optimizer_learning_rate = model_config["no_transfer_lstm"][
            "optimizer_learning_rate"
        ]  # Learning rate for optimizer
        self.amsgrad = model_config["no_transfer_lstm"][
            "amsgrad"
        ]  # Use AMSGrad variant of Adam
        self.export_weights = model_config["no_transfer_lstm"][
            "export_model"
        ]  # Whether to export weights
        self.n_hi = model_config["no_transfer_lstm"]["out_n_hi"]  # Hidden size for LSTM
        self.n_layers = model_config["no_transfer_lstm"][
            "n_layers"
        ]  # Number of LSTM layers

        # Prepare lists of tasks and sub-tasks
        self.multy_task_learning_list = list(
            self.Xtrain_tasks.keys()
        )  # Top-level tasks
        self.sub_multy_task_learning_list = {}  # Dict to hold sub-tasks for each task

        # Dictionaries to hold models, optimizers, and losses for each (task, sub_task)
        (
            self.model_lin_dict,  # Linear layers for each (task, sub_task)
            self.model_lstm_dict,  # LSTM models for each (task, sub_task)
            self.opt_dict,  # Optimizers for each (task, sub_task)
            self.signal_layer,  # Activation layers for each (task, sub_task)
            self.losses,  # Loss history for each (task, sub_task)
        ) = {}, {}, {}, {}, {}

        # Loop over each task
        for task in self.multy_task_learning_list:
            # Pre-allocate dictionaries for each task
            (
                self.model_lin_dict[task],
                self.signal_layer[task],
                self.model_lstm_dict[task],
                self.opt_dict[task],
                self.losses[task],
            ) = {}, {}, {}, {}, {}

            # Get the list of sub-tasks for this task
            self.sub_multy_task_learning_list[task] = list(
                self.Xtrain_tasks[task].keys()
            )

            # Loop over each sub-task
            for sub_task in self.sub_multy_task_learning_list[task]:
                # Initialize loss history for this (task, sub_task)
                self.losses[task][sub_task] = []
                # Number of input features (assume input and output dims are the same)
                n_in = self.Xtrain_tasks[task][sub_task].shape[1]
                n_out = self.Xtrain_tasks[task][sub_task].shape[1]

                # Set hidden size and number of layers for LSTM
                n_hi, n_layers = self.n_hi, self.n_layers

                # Create LSTM model for this (task, sub_task)
                self.model_lstm_dict[task][sub_task] = (
                    nn.LSTM(n_in, n_hi, n_layers, batch_first=True)
                    .double()
                    .to(self.device)
                )
                # Create linear output layer for this (task, sub_task)
                self.model_lin_dict[task][sub_task] = (
                    nn.Linear(n_hi, n_out).double().to(self.device)
                )
                # Create activation layer (Tanh) for this (task, sub_task)
                self.signal_layer[task][sub_task] = nn.Tanh().to(self.device)

                # Create optimizer for this (task, sub_task), including all model parameters
                self.opt_dict[task][sub_task] = torch.optim.Adam(
                    list(self.model_lstm_dict[task][sub_task].parameters())
                    + list(self.model_lin_dict[task][sub_task].parameters())
                    + list(self.signal_layer[task][sub_task].parameters()),
                    lr=self.optimizer_learning_rate,
                    amsgrad=self.amsgrad,
                )
                # Print model and optimizer details for debugging
                print(
                    task,
                    sub_task,
                    self.model_lstm_dict[task][sub_task],
                    self.model_lin_dict[task][sub_task],
                    self.signal_layer[task][sub_task],
                    self.opt_dict[task][sub_task],
                )

    def train(self):
        # Main training loop for the specified number of training steps
        for i in range(self.train_steps):
            # Loop over each main task
            for task in self.multy_task_learning_list:
                # Loop over each sub-task for the current main task
                for sub_task in self.sub_multy_task_learning_list[task]:
                    # Randomly select batch start indices for this sub-task
                    start_ids = np.random.permutation(
                        list(
                            range(
                                self.Xtrain_tasks[task][sub_task].size(0)
                                - self.seq_len
                                - 1
                            )
                        )
                    )[: self.batch_size]
                    # Stack batches of sequences for training
                    XYbatch = torch.stack(
                        [
                            self.Xtrain_tasks[task][sub_task][i : i + self.seq_len + 1]
                            for i in start_ids
                        ],
                        dim=0,
                    )
                    # Ytrain: target values (one-step ahead for each sequence in the batch)
                    Ytrain = XYbatch[:, 1:, :]
                    # Xtrain: input values (all but last step in each sequence in the batch)
                    Xtrain = XYbatch[:, :-1, :]

                    # Reset gradients for the optimizer before backpropagation
                    self.opt_dict[task][sub_task].zero_grad()

                    # Initialize hidden and cell states for LSTM for this batch
                    (hidden, cell) = self.get_hidden(
                        self.batch_size, self.n_layers, self.n_hi
                    )
                    # Forward pass through LSTM
                    hidden, cell = self.model_lstm_dict[task][sub_task](
                        Xtrain, (hidden, cell)
                    )
                    # Pass LSTM output through linear and activation layers to get predictions
                    preds = self.signal_layer[task][sub_task](
                        self.model_lin_dict[task][sub_task](hidden)
                    )

                    # Compute loss using the specified criterion (e.g., Sharpe ratio loss)
                    loss = self.criterion(preds, Ytrain)
                    # Store the loss value for monitoring
                    self.losses[task][sub_task].append(loss.item())

                    # Backpropagation: compute gradients
                    loss.backward()
                    # Update model parameters using optimizer
                    self.opt_dict[task][sub_task].step()

            # Print progress every 100 iterations (when i mod 100 == 1)
            if (i % 100) == 1:
                print(i)

        # Optionally export model weights after training if export_weights is True
        if self.export_weights:
            for task in self.multy_task_learning_list:
                for sub_task in self.sub_multy_task_learning_list[task]:
                    # Save the linear layer weights for this (task, sub_task)
                    torch.save(
                        self.model_lin_dict[task][sub_task],
                        self.export_path
                        + task
                        + "_"
                        + sub_task
                        + "_"
                        + self.export_label
                        + "_linnotransferlstm.pt",
                    )
                    # Save the LSTM model weights for this (task, sub_task)
                    torch.save(
                        self.model_lstm_dict[task][sub_task],
                        self.export_path
                        + task
                        + "_"
                        + sub_task
                        + "_"
                        + self.export_label
                        + "_notransferlstm.pt",
                    )

    def predict(self, x_test):
        """
        Generate predictions for the given test data using the trained LSTM and linear layers.

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
                    # Initialize hidden and cell states for LSTM
                    (hidden, cell) = self.get_hidden(1, self.n_layers, self.n_hi)
                    # Forward pass through LSTM (excluding the last time step)
                    hidden, _ = self.model_lstm_dict[task][sub_task](
                        xflat[:, :-1], (hidden, cell)
                    )
                    # Pass LSTM output through linear and activation layers to get predictions
                    y_pred[task][sub_task] = self.signal_layer[task][sub_task](
                        self.model_lin_dict[task][sub_task](hidden)
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
        filename = f"no_transfer_lstm_trainsteps_{self.train_steps}.pkl.gz"
        filepath = os.path.join(folder_path, filename)

        model_state = {
            "model_lin_dict": self.model_lin_dict,
            "model_lstm_dict": self.model_lstm_dict,
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
            "n_hi": self.n_hi,
            "n_layers": self.n_layers,
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
        instance.model_lin_dict = model_state["model_lin_dict"]
        instance.model_lstm_dict = model_state["model_lstm_dict"]
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
        instance.n_hi = model_state["n_hi"]
        instance.n_layers = model_state["n_layers"]

        # Set the criterion function
        instance.criterion = instance.avg_sharpe_ratio

        # Note: Xtrain_tasks is not saved as it's not needed for prediction
        # It was only used during training for data access

        return instance
