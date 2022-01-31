import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class ESN():

    def __init__(self, n_inputs, n_outputs, reservoir_size=200,
                 leaking_rate=1.0, spectral_radius=1.0, washout_time=0,
                 starting_state='zeros', noise=0,
                 ridge_param=0, W_in_scaling=1.0, W_scaling=1.0,
                 W_fb_scaling=1.0, bias_scaling=1.0, activation_func='tanh',
                 ive=None, ove=None, random_state=None, silent=True):
        """
        Args:
            n_inputs: nr of input dimensions
            n_outputs: nr of output dimensions
            reservoir_size: N dimension of the N x N reservoir
            leaking_rate: The speed of the reservoir update dynamics discretized in time
            spectral_radius: spectral radius of the recurrent weight matrix
            washout_time: First N state vectors x that need to be discared during training
            starting_state: start internal state with random vector or zero vector (zeros/random)
            noise: noise added to each neuron (regularization)
            ridge_param: Regularization strength of the ridge regression
            W_in_scaling: Scaling of the Input wight matrix (single scalar)
            W_scaling: Scaling of the reservoir connections (single scalar)
            W_fb: Scaling of the feeback weights (single scalar)
            W_bias: Scalaing of the bias vector (single scalar)
            ive: Input vector encoder
            ove: Output vector encoder
            random_state: positive integer seed
            silent: Surpress messages
        """

        # check for proper dimensionality of all arguments and write them down.
        self.n_inputs = n_inputs
        self.reservoir_size = reservoir_size
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.washout_time = washout_time
        self.leaking_rate = leaking_rate
        self.starting_state = starting_state
        self.noise = noise
        self.ridge_param = ridge_param
        self.W_in_scaling = W_in_scaling
        self.W_scaling = W_scaling
        self.W_fb_scaling = W_fb_scaling
        self.bias_scaling = bias_scaling
        self.activation_func = activation_func

        if random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: " + str(e))
        else:
            self.random_state_ = np.random.mtrand._rand

        self.ive = ive
        self.ove = ove
        self.silent = silent
        self.now = datetime.now()

        # Initialize the weight
        self._init_weights()

    def _init_weights(self):
        """
        Method to initialize the weight matrices W_in, W, W_fb, vector b, and internal stat vector x
        All tensors entries are sampled from the uniform distribution [-0.5, 0.5]
        Scale them by requested amount

        W_in: reservoir_size x n_inputs
        W: reservoir_size x reservoir_size
        W_fb: reservoir_size x n_outputs
        bias: reservoir_size
        """
        self.W_in = self.random_state_.rand(self.reservoir_size, self.n_inputs) - 0.5

        # begin with a random matrix centered around zero:
        W = self.random_state_.rand(self.reservoir_size, self.reservoir_size) - 0.5
        # rescale them to reach the requested spectral radius:
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (self.spectral_radius / radius)

        self.W_fb = self.random_state_.rand(self.reservoir_size, self.n_outputs) - 0.5
        self.bias = self.random_state_.rand(self.reservoir_size) - 0.5

        # Scale all tensors
        self.W_in *= self.W_in_scaling
        self.W *= self.W_scaling
        self.W_fb *= self.W_fb_scaling
        self.bias *= self.bias_scaling

        # Initialize starting state
        self._init_internal_state(self.starting_state)

    def _init_internal_state(self, values):
        """
        Method to initialize the internal state
        Args:
            values: String the specify whether state should be initialized randomly, or with zeros
        """
        if values == 'zeros':
            self.x = np.zeros(self.reservoir_size)
        else:
            self.x = self.random_state_.rand(self.reservoir_size) - 0.5

    def _activation_function(self, x):
        """
        Method to apply activation function
        """
        if self.activation_func == 'tanh':
            x = np.tanh(x)
        if self.activation_func == 'sigmoid':
            x = 1/(1 + np.exp(-x))
        return x

    def _update(self, u, y):
        """
        Method to perform update step and sets self.x to self.x = x(n+1)
        Args:
            input u(n+1)
            output y(n)
        Network is governed by update equation:
        x(n+1) = (1-leaking_rate)x(n) + leaking_rate*f(W_in*u(n+1) + W*x(n) + W_fb*y(n-1) + bias)
        """
        pre_x = np.dot(self.W_in, u) + np.dot(self.W, self.x) + np.dot(self.W_fb, y) + self.bias
        pre_x = self.leaking_rate * self._activation_function(pre_x)
        new_x = (1 - self.leaking_rate)*self.x + pre_x
        self.x = new_x

    def _harvest_states(self, u_train, y_teacher):
        """
        Method to harvest states used to determine W_out
        Args:
            u_inputs: np.array of dimensions (N_training_samples x n_inputs) with training inputs
            y_teacher: np.array of dimension (N_training_samples x n_outputs) with teacher outputs
        Returns:
            states: matrix of size N_train_samples x reservoir_size, containing the states it its rows
        """
        # Initialize internal state x (as a zero vector) for every training input
        n_train_samples = u_train.shape[0]
        states = np.zeros((n_train_samples, self.reservoir_size))

        # Harvest states by driving the network with the given input/output pairs
        # Drive the network once where y[n-1] does no exists
        u = u_train[0, :]
        y = np.zeros(self.n_outputs)
        self._update(u, y)
        states[0, :] = self.x

        for n in range(1, n_train_samples):
            u = u_train[n, :]
            y = y_teacher[n-1, :]
            self._update(u, y)
            states[n, :] = self.x
        return states

    def _add_bias(self, x):
        """
        Method to add bias input to a vector or matrix
        Args:
            x: Vector as an np.array of
        Returns:
            x_padded: vector that has all 1s as an extra column
        """
        return np.c_[x, np.ones(x.shape[0])]

    def _ridge_regression(self, X, D):
        """
        Method to apply ridge regression
        W_out = (X'X + a^2I)^-1 X'D
        Args:
            Matrix X of size N_training_samples x reservoir_size
            Matrix D of zie N_training_samples x n_outputs
        Returns:
            Optimal weight matrix W_out of dimensions: n_outputs x (reservoir_size + 1 (bias))
        """
        # Add bias of constant 1 to X matrix (as a column)
        X = self._add_bias(X)
        n = X.shape[1]
        R = np.dot(X.T, X)
        P = np.dot(X.T, D)
        W_out = np.dot(np.linalg.inv(R + (self.ridge_param**2)*np.identity(n)), P)
        # transpose matrix to fit dimensions n_outputs x reservoir_size
        return W_out.T

    def _compute_mse(self, y, y_pred):
        """
        Method to compute the mean square error using weight matrix W_out
        Args:
            Matrix y_ped of size N_training_samples x n_outputs
            Matrix y of size N_training_samples x n_outputs
        Returns:
            Mean square error
        """
        return np.sqrt(np.mean((y_pred - y)**2))

    def _acc_score(self, y, y_pred):
        """
        Method to compute accuracy score of how many entries in y and y_pred are equal and returns
        them averaged
        Args:
            y: real vector y
            y_pred: vector with predicted values
            ove: Output vector encoder
        Returns:
            Averged number of entries where real midi values == predicted midi values
        """
        real_midi = self.ove.inv_transform_max_probability(y)
        pred_midi = self.ove.inv_transform_max_probability(y_pred)
        return np.mean(real_midi == pred_midi)

    def _save_states(self, states):
        """
        Method to save numpy array to folder 'states/'
        Args:
            Numpy array
        """
        dt_string = self.now.strftime("%d-%m-%Y %H:%M:%S")
        np.save(f'states/states_from_{dt_string}', states)

    def fit(self, u_train, y_teacher, save_states=False):
        """
        Method to find optimal weights for W_out using ridge regression
        Args:
            u_inputs: np.array of dimensions (N_training_samples x n_inputs) with training inputs
            y_teacher: np.array of dimension (N_training_samples x n_outputs) with teacher outputs
            save_states: Bool to specify wether generated states should be saved (determining washout time)
        Returns:
            The accuracy and mean square error on the training data
        """
        self._log('Harvesting states...')
        states = self._harvest_states(u_train, y_teacher)

        if save_states:
            self._save_states(states)

        # Discard states[n < washout_time] and correspoding y_teachers[n < washout_time]
        states = states[self.washout_time:, :]
        y_teacher = y_teacher[self.washout_time:, :]

        # Learn the weights of W_out using ridge regresion
        self._log("Fitting...")
        self.W_out = self._ridge_regression(states, y_teacher)

        # Compute training MSE
        X = self._add_bias(states)
        y_pred = np.dot(X, self.W_out.T)
        train_mse = self._compute_mse(y_teacher, y_pred)
        train_acc = self._acc_score(y_teacher, y_pred)
        self._log(f'Finished training! With train acc: {train_acc:.3f} and MSE: {train_mse}')
        return train_acc, train_mse

    def determine_washout_time(self, u_train, y_teacher, n_neurons, max_t):
        """
        Mehtod to determine the washout time. Drives the network twice and plots the
        neuron activations
        Args:
            u_inputs: np.array of dimensions (N_training_samples x n_inputs) with training inputs
            y_teacher: np.array of dimension (N_training_samples x n_outputs) with teacher outputs
            n_neurons: int to specify the number of neurons to plot
            max_t: int to specify the time interval of plotting 0, max_t
        """
        # Drive the network twice with different internal state x
        self._init_internal_state('random')
        states_1 = self._harvest_states(u_train, y_teacher)

        self._init_internal_state('random')
        states_2 = self._harvest_states(u_train, y_teacher)

        # Initialize arrays
        data = np.empty((2*n_neurons*max_t, 4))
        time = []
        values = []
        neuron_idx = []
        run = []

        t = list(range(0, max_t))
        run_1 = [1 for i in range(max_t)]
        run_2 = [2 for i in range(max_t)]

        for i in range(n_neurons):
            time = time + t + t

            values_1 = states_1[0:max_t, i]
            values_2 = states_2[0:max_t, i]
            values = values + values_1.tolist() + values_2.tolist()

            idxs = [i for j in range(max_t)]
            neuron_idx = neuron_idx + idxs + idxs

            run = run + run_1 + run_2

        data[:, 0] = time
        data[:, 1] = values
        data[:, 2] = neuron_idx
        data[:, 3] = run
        df = pd.DataFrame(data, columns=['time', 'values', 'neuron_idx', 'run'])
        sns.lineplot(data=df, x='time', y='values', hue='neuron_idx', style='run')
        plt.show()

    def validate(self, u_val, y_val):
        """
        Method to test the prediction performance of the network by driving the network with input u_val
        and comparing the output of the network to the ground truth vectors y_val. It assumed that the validation
        set is the last X% of the training set (thus internal state x is not changed)
        Args:
            u_inputs: np.array of dimensions (N_samples x n_inputs)
            y_teacher: np.array of dimension (N_samples x n_outputs)
        Returns:
            The prediction accuracy
        """
        self._log('Validating...')
        acc = 0.0
        n_val_examples = u_val.shape[0]
        y_pred = np.empty((n_val_examples, self.n_outputs))

        # Drive the network once where y[n-1] does no exists
        u = u_val[0, :]
        y0 = np.zeros(self.n_outputs)
        self._update(u, y0)
        x_n = self._add_bias(self.x.reshape((1, -1)))
        y_pred[0, :] = np.dot(self.W_out, x_n.T).reshape(-1)

        for n in range(1, n_val_examples):
            u_n = u_val[n, :]
            y_n0 = y_pred[n-1, :]
            # Get x(n)
            self._update(u_n, y_n0)
            # Compute y(n)
            x_n = self._add_bias(self.x.reshape((1, -1)))
            y_pred[n, :] = np.dot(self.W_out, x_n.T).reshape(-1)

        acc = self._acc_score(y_val, y_pred)
        self._log(f'Validation accuracy = {acc:.3f}')
        return acc

    def predict_sequence(self, u_drive, y_drive, l):
        """
        Method to predict a sequence
        Args:
            u_drive: Number of input sequences to drive (start) the networks
            y_drive: Teacher outputs to drive the network
            l: Length of the requested sequence
        Returns:
            A matrix of size l x output dimensions
        """

        # Add one to l to make up for loop starting at 1:l
        l += 1
        # drive the network
        _ = self._harvest_states(u_drive, y_drive)

        # Setup empty matrices
        pred_sequence = np.empty((l, self.ove.n_channels))
        u = np.empty((l+1, self.n_inputs))
        y = np.empty((l, self.n_outputs))

        # Get the last teacher vector and set it as y(n-1)
        y[0, :] = y_drive[-1, :]
        # Convert y(n-1) to u(n)
        out = self.ove.inv_transform_max_probability(np.array([y[0, :]]))[0]
        u[1, :] = self.ive.transform(np.array([out]))

        for n in range(1, l):
            u_n = u[n, :]
            y_n0 = y[n-1, :]
            # Get x(n)
            self._update(u_n, y_n0)
            # Compute y(n)
            x_n = self._add_bias(self.x.reshape((1, -1)))
            y_n = np.dot(self.W_out, x_n.T)
            # Convert y(n) to a midi value, and back into u(n+1)
            out = self.ove.inv_transform_max_probability(y_n.T)[0]
            u[n+1, :] = self.ive.transform([out])

            # Save predicted midi values
            pred_sequence[n, :] = out

        return pred_sequence[1:, :].astype(int)

    def save_model(self, ):
        """
        Method to save model in the folder '/models'
        """
        models_dir = Path.cwd() / 'models'
        if not models_dir.is_dir():
            os.makedirs(models_dir)

        dt_string = self.now.strftime("%d-%m-%Y %H:%M:%S")
        save_dir = models_dir / dt_string
        os.makedirs(save_dir)

        np.save(save_dir / 'W_in', self.W_in)
        np.save(save_dir / 'W', self.W)
        np.save(save_dir / 'W_fb', self.W_fb)
        np.save(save_dir / 'W_out', self.W_out)
        np.save(save_dir / 'bias', self.bias)

    def _log(self, msg):
        """
        Method to print messages when self.silent=False
        """
        if not self.silent:
            print(msg)
