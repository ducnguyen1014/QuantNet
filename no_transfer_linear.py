import gzip
import os
import pickle

import numpy as np
import torch
from torch import nn


class NoTransferLinear:
    def __init__(self, x_tasks, model_config):
        # Set the loss function to be used during training (here, average Sharpe ratio)
        self.criterion = self.avg_sharpe_ratio  # Alternative: nn.MSELoss().cuda()
        # Store the training data for all tasks
        self.Xtrain_tasks = x_tasks
        # Number of total training steps for all tasks
        self.train_steps = model_config["train_steps"]
        # Number of training steps per task (if used)
        self.tasks_train_steps = model_config["tasks_train_steps"]
        # Batch size for training
        self.batch_size = model_config["batch_size"]
        # Sequence length for each training sample
        self.seq_len = model_config["seq_len"]
        # Device to run the model on (e.g., 'cuda' or 'cpu')
        self.device = model_config["device"]
        # Path to export trained model weights
        self.export_path = model_config["export_path"]
        # Label to use when exporting model weights
        self.export_label = model_config["export_label"]

        # Model-specific parameters for the no_transfer_linear model
        self.optimizer_learning_rate = model_config["no_transfer_linear"][
            "optimizer_learning_rate"
        ]  # Learning rate for optimizer
        self.amsgrad = model_config["no_transfer_linear"][
            "amsgrad"
        ]  # Whether to use AMSGrad variant of Adam
        self.export_weights = model_config["no_transfer_linear"][
            "export_weights"
        ]  # Whether to export weights after training

        # Prepare lists and dictionaries to manage tasks and sub-tasks
        self.multy_task_learning_list = list(
            self.Xtrain_tasks.keys()
        )  # List of main tasks
        self.sub_multy_task_learning_list = {}  # Dict to hold sub-tasks for each main task
        # Dictionaries to hold models, optimizers, activation layers, and loss histories for each (task, sub-task)
        self.model_lin_dict, self.opt_dict, self.signal_layer, self.losses = (
            {},
            {},
            {},
            {},
        )
        # Iterate over each main task
        for task in self.multy_task_learning_list:
            # Initialize dictionaries for each main task
            (
                self.model_lin_dict[task],
                self.signal_layer[task],
                self.opt_dict[task],
                self.losses[task],
            ) = {}, {}, {}, {}
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

                # Create a linear model for this (task, sub-task), move to the specified device, and use double precision
                self.model_lin_dict[task][sub_task] = (
                    nn.Linear(n_in, n_out).double().to(self.device)
                )
                # Add a Tanh activation layer for this (task, sub-task)
                self.signal_layer[task][sub_task] = nn.Tanh().to(self.device)

                # Create an Adam optimizer for the parameters of both the linear and activation layers
                self.opt_dict[task][sub_task] = torch.optim.Adam(
                    list(self.model_lin_dict[task][sub_task].parameters())
                    + list(self.signal_layer[task][sub_task].parameters()),
                    lr=self.optimizer_learning_rate,
                    amsgrad=self.amsgrad,
                )
                # Print out the configuration for this (task, sub-task) for debugging/verification
                print(
                    task,
                    sub_task,
                    self.model_lin_dict[task][sub_task],
                    self.signal_layer[task][sub_task],
                    self.opt_dict[task][sub_task],
                )

    def train(self):
        # Loop over the total number of training steps
        for i in range(self.train_steps):
            # Iterate over each main task
            for task in self.multy_task_learning_list:
                # Iterate over each sub-task for the current main task
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

                    # Forward pass: compute predictions using the linear model and activation layer
                    preds = self.signal_layer[task][sub_task](
                        self.model_lin_dict[task][sub_task](Xtrain)
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
                    torch.save(
                        self.model_lin_dict[task][sub_task],
                        self.export_path
                        + task
                        + "_"
                        + sub_task
                        + "_"
                        + self.export_label
                        + "_notransferlinear.pt",
                    )

    def predict(self, x_test):
        y_pred = {}
        for task in self.multy_task_learning_list:
            y_pred[task] = {}
            for sub_task in self.sub_multy_task_learning_list[task]:
                # we still need a batch dim, but it's just 1
                xflat = x_test[task][sub_task].view(
                    1, -1, x_test[task][sub_task].size(1)
                )
                with torch.autograd.no_grad():
                    y_pred[task][sub_task] = self.signal_layer[task][sub_task](
                        self.model_lin_dict[task][sub_task](xflat[:, :-1])
                    )

        return y_pred

    def avg_sharpe_ratio(self, output, target):
        # Set slippage and basis point cost to zero (can be adjusted if needed)
        slip = 0.0005 * 0.00
        bp = 0.0020 * 0.00

        # Calculate returns: element-wise multiplication of model output and target
        rets = torch.mul(output, target)

        # Calculate transaction costs:
        # Compute the absolute difference between consecutive outputs (positions)
        # Multiply by total transaction cost per trade (bp + slip)
        tc = torch.abs(output[:, 1:, :] - output[:, :-1, :]) * (bp + slip)

        # Prepend a zero transaction cost for the first time step (no previous position)
        tc = torch.cat(
            [
                torch.zeros(output.size(0), 1, output.size(2)).double().to(self.device),
                tc,
            ],
            dim=1,
        )

        # Subtract transaction costs from returns
        rets = rets - tc

        # Compute average returns across all elements
        avg_rets = torch.mean(rets)

        # Compute standard deviation (volatility) of returns
        vol_rets = torch.std(rets)

        # Compute negative Sharpe ratio (for minimization in optimization)
        loss = torch.neg(torch.div(avg_rets, vol_rets))

        # Return the mean loss (in case of batch dimension)
        return loss.mean()

    def save_model(self, folder_path):
        """
        Save the entire model state as a pickle file
        """
        # Ensure the folder exists
        os.makedirs(folder_path, exist_ok=True)

        # Generate filename with training steps
        filename = f"no_transfer_linear_trainsteps_{self.train_steps}.pkl.gz"
        filepath = os.path.join(folder_path, filename)

        model_state = {
            "model_lin_dict": self.model_lin_dict,
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

        # Set the criterion function
        instance.criterion = instance.avg_sharpe_ratio

        # Note: Xtrain_tasks is not saved as it's not needed for prediction
        # It was only used during training for data access

        return instance
