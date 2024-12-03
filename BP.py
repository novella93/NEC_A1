import numpy as np

# Activation function (Sigmoid) and its derivative for the forward and backward propagation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNet:
    def __init__(self, layers, epochs, learning_speed, momentum):
        ############################
        # ATTRIBUTE INITIALIZATION #
        ############################

        # Number of layers
        self.L = len(layers)
        # Layers
        self.n = layers.copy()
        # Epochs
        self.epochs = epochs
        # Learning speed
        self.learning_speed = learning_speed
        # Momentum
        self.momentum = momentum

        # Fields
        self.h = [np.zeros((n, 1)) for n in layers]

        # Activations of the networks
        self.xi = [np.zeros((n, 1)) for n in layers]

        # Weights
        self.w = [np.random.randn(layers[i], layers[i - 1]) for i in range(1, self.L)]
        # Weights changes
        self.d_w = [np.zeros((layers[i], layers[i - 1])) for i in range(1, self.L)]
        # Previous changes for weights
        self.d_w_prev = [np.zeros((layers[i], layers[i - 1])) for i in range(1, self.L)]

        # Thresholds
        self.theta = [np.random.randn(n, 1) for n in layers[1:]]
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
        activations = [X]
        for i in range(len(self.w)):
            X = sigmoid(np.dot(self.w[i], X) + self.theta[i])
            activations.append(X)
        return activations
    
    def backward_propagation(self, y):
        # Compute the gradient of the loss with respect the 
        self.delta[-1] = (self.xi[-1] - y) * sigmoid_derivative(self.h[-1])
        for i in range(self.L - 2, 0, -1):
            self.delta[i] = np.dot(self.w[i].T, self.delta[i + 1]) * sigmoid_derivative(self.h[i])
        for i in range(self.L - 1):
            self.d_w[i] = np.dot(self.delta[i + 1], self.xi[i].T)
            self.d_theta[i] = self.delta[i + 1]

            # Update weights and thresholds with momentum
            self.w[i] -= self.learning_speed * self.d_w[i] + self.momentum * self.d_w_prev[i]
            self.theta[i] += self.learning_speed * self.d_theta[i] + self.momentum * self.d_theta_prev[i]

            # Store current changes for the momentum term
            self.d_w_prev[i] = self.d_w[i]
            self.d_theta_prev[i] = self.d_theta[i]

    def train(self, X, y):
        # Iterate over the epochs for the dataset
        for epoch in range(self.epochs):
            # Iterate each training sample for forward and backward passes
            for j in range(X.shape[1]):
                # Perform a forward propagation and reshape it as column vector
                self.forward_propagation(X[:, j].reshape(-1, 1))
                # Perform backward propagation and reshape it as column vector
                self.backward_propagation(y[:, j].reshape(-1, 1))
            # Monitor training process
            if epoch % 1000 == 0:
                loss = np.mean((y - self.forward_propagation(X)[-1]) ** 2)
                print('Epoch number: ' + str(epoch) + ', Loss value: ' + str(loss))

    def predict(self, X):
        return self.forward_propagation(X)[-1]

# Data preprocessing (scaling) 
def scale_data(X, min_val, max_val): 
    return (X - min_val) / (max_val - min_val)
    
# Example usage
if __name__ == "__main__":
    # Example data (X: inputs, y: targets)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
    y = np.array([[0], [1], [1], [0]]).T

    # Scale data
    X_scaled = scale_data(X, X.min(), X.max())

    # Initialize and train the neural network
    nn = NeuralNet(layers=[2, 3, 1], epochs=10000, learning_speed=0.1, momentum=0.9)
    nn.train(X_scaled, y)

    # Make predictions
    predictions = nn.predict(X_scaled)
    print(f'Predictions:\n{predictions}')