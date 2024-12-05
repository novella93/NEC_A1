import numpy as np

############################
#   ACTIVATION FUNCTIONS   #
############################

# Define sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define relu activation function and its derivative
def relu(x): 
    return np.maximum(0, x) 

def relu_derivative(x): 
    return np.where(x > 0, 1, 0)

# Define leaky relu activation function and its derivative
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

# Define linear activation function and its derivative
def linear(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)

# Define tanh activation function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


class NeuralNet:
    def __init__(self, layers, epochs, learning_rate, momentum, fact):
        ############################
        # ATTRIBUTE INITIALIZATION #
        ############################

        # Number of layers
        self.L = len(layers)
        # Layers
        self.n = layers.copy()
        # Epochs
        self.epochs = epochs
        # Learning rate
        self.learning_rate = learning_rate
        # Momentum
        self.momentum = momentum
        # Fact
        self.fact = fact

        # Fields
        self.h = [np.zeros((n, 1)) for n in layers]

        # Activations of the networks
        self.xi = [np.zeros((n, 1)) for n in layers]

        # Weights (with smaller initialization)
        # self.w = [np.random.randn(layers[i], layers[i - 1]) * 0.01 for i in range(1, self.L)]
        # Xavier initialization
        self.w = [np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2 / (layers[i] + layers[i - 1])) for i in range(1, self.L)]
        # Weights changes
        self.d_w = [np.zeros((layers[i], layers[i - 1])) for i in range(1, self.L)]
        # Previous changes for weights
        self.d_w_prev = [np.zeros((layers[i], layers[i - 1])) for i in range(1, self.L)]

        # Thresholds
        # self.theta = [np.random.randn(n, 1) * 0.01 for n in layers[1:]]
        # Xavier intialization
        self.theta = [np.random.randn(n, 1) * np.sqrt(2 / (layers[i])) for i, n in enumerate(layers[1:])]
        # Thresholds changes
        self.d_theta = [np.zeros((n, 1)) for n in layers[1:]]
        # Previous changes for thresholds
        self.d_theta_prev = [np.zeros((n, 1)) for n in layers[1:]]

        # Propagation of error
        self.delta = [np.zeros((n, 1)) for n in layers]

    #####################
    # METHOD DEFINITION #
    #####################

    def forward_propagation(self, X):
        self.xi[0] = X
        for i in range(1, self.L):
            self.h[i] = np.dot(self.w[i - 1], self.xi[i - 1]) - self.theta[i - 1]
            self.xi[i] = sigmoid(self.h[i])
        return self.xi
    
    def backward_propagation(self, y):
        # Compute the gradient of the loss with respect to the output
        self.delta[-1] = (self.xi[-1] - y) * sigmoid_derivative(self.h[-1])
        for i in range(self.L - 2, 0, -1):
            self.delta[i] = np.dot(self.w[i].T, self.delta[i + 1]) * sigmoid_derivative(self.h[i])
        for i in range(self.L - 1):
            self.d_w[i] = np.dot(self.delta[i + 1], self.xi[i].T)
            self.d_theta[i] = self.delta[i + 1]

            # Gradient clipping
            np.clip(self.d_w[i], -1, 1, out=self.d_w[i])
            np.clip(self.d_theta[i], -1, 1, out=self.d_theta[i])

            # Update weights and thresholds with momentum
            self.w[i] -= self.learning_rate * self.d_w[i] + self.momentum * self.d_w_prev[i]
            self.theta[i] -= self.learning_rate * self.d_theta[i] + self.momentum * self.d_theta_prev[i]

            # Store current changes for the momentum term
            self.d_w_prev[i] = self.d_w[i]
            self.d_theta_prev[i] = self.d_theta[i]

    def train(self, X, y):
        # Iterate over the epochs for the dataset
        for epoch in range(self.epochs):
            # Iterate each training sample for forward and backward passes
            for j in range(X.shape[1]):
                # Perform a forward propagation and reshape it as column vector
                self.forward_propagation(X[j, :].reshape(-1, 1))
                # Perform backward propagation and reshape it as column vector
                self.backward_propagation(y[j, :].reshape(-1, 1))
            # Monitor training process
            if epoch % 1000 == 0:
                predictions = np.hstack([self.forward_propagation(X[j, :].reshape(-1, 1))[-1] for j in range(X.shape[0])])
                loss = np.mean((y - predictions) ** 2)
                print(f'Epoch number: {epoch}, Loss value: {loss}')

    def predict(self, X):
        return np.hstack([self.forward_propagation(X[j, :].reshape(-1, 1))[-1] for j in range(X.shape[0])])

# Data preprocessing (scaling) 
def scale_data(X, min_val, max_val): 
    return (X - min_val) / (max_val - min_val)