import numpy as np
from datetime import datetime


class ESN():

    def __init__(self, n_inputs, n_outputs, reservoir_size=200,
                 leaking_rate=1.0, spectral_radius=1.0, washout_time=0,
                 starting_state='zeros', noise=0,
                 ridge_param=0, W_in_scaling=1.0, W_scaling=1.0,
                 W_fb_scaling=1.0, bias_scaling=1.0, activation_func='tanh',
                 random_state=None, silent=True):
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

        self.silent = silent

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

        if self.starting_state == 'zeros':
            self.x = np.zeros(self.reservoir_size)
        else:
            self.x = self.random_state_.rand(self.reservoir_size) - 0.5

        # Scale all tensors
        self.W_in *= self.W_in_scaling
        self.W *= self.W_scaling
        self.W_fb *= self.W_fb_scaling
        self.bias *= self.bias_scaling

    def _activation_function(self, x):
        """
        Method to apply activation function
        """
        if self.activation_func == 'tanh':
            x = np.tanh(x)
        return x

    def _update(self, u, y):
        """ 
        Method to perform update step and sets self.x to self.x = x(n+1)
        Args:
            input u(n+1)
            output y(n)
        Network is governed by update equation:
        x(n+1) = (1-leaking_rate)x(n) + leaking_rate*f(W_in*u(n+1) + W*x(n) + W_fb*y(n) + bias)
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

    def _ridge_regression(self, X, D):  # @TODO does this need padding with const 1?
        """
        Method to apply ridge regression
        W_out = (X'X + a^2I)^-1 X'D
        Args:
            Matrix X of size N_training_samples x reservoir_size
            Matrix D of zie N_training_samples x n_outputs
        Returns:
            Optimal weight matrix W_out of dimensions: n_outputs x reservoir_size
        """
        n = X.shape[1]
        R = np.dot(X.T, X)
        P = np.dot(X.T, D)
        W_out = np.dot(np.linalg.inv(R + (self.ridge_param**2)*np.identity(n)), P)
        # transpose matrix to fit dimensions n_outputs x reservoir_size
        return W_out.T

    def _compute_mse(self, X, y):
        """
        Method to compute the mean square error using weight matrix W_out
        Args:
            Matrix X of size N_training_samples x reservoir_size
            Matrix y of size N_training_samples x n_outputs
        Returns:
            Mean square error
        """
        predictions = np.dot(X, self.W_out.T)
        return np.sqrt(np.mean((predictions - y)**2))

    def _save_states(self, states):
        """
        Method to save numpy array to folder 'states/'
        Args:
            Numpy array
        """
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
        np.save(f'states/states_from_{dt_string}', states)

    def fit(self, u_train, y_teacher, save_states=False):
        """
        Method to find optimal weights for W_out using ridge regression
        Args:
            u_inputs: np.array of dimensions (N_training_samples x n_inputs) with training inputs
            y_teacher: np.array of dimension (N_training_samples x n_outputs) with teacher outputs
            save_states: Bool to specify wether generated states should be saved (determining washout time)
        Returns:
            The mean square error on the training data
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
        train_mse = self._compute_mse(states, y_teacher)
        self._log(f'Finished training! With train MSE: {train_mse}')
        return train_mse

    def _log(self, msg):
        """
        Method to print messages when self.silent=False
        """
        if not self.silent:
            print(msg)
