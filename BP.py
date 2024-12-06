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

        # Fields initialized with batch size of 1
        self.h = [np.zeros((1, layers[i])) for i in range(self.L)]

        # Activations of the networks initialized with batch size of 1
        self.xi = [np.zeros((1, layers[i])) for i in range(self.L)]

        # Weights (with smaller initialization)
        # self.w = [np.random.randn(layers[i], layers[i - 1]) * 0.01 for i in range(1, self.L)]
        # Xavier initialization
        # self.w = [np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2 / (layers[i] + layers[i - 1])) for i in range(1, self.L)]
        # He initialization
        self.w = [np.random.randn(layers[i], layers[i+1]) * np.sqrt(2. / layers[i]) for i in range(self.L - 1)]
        
        # Weights changes
        self.d_w = [np.zeros_like(self.w[i]) for i in range(self.L-1)]
        # Previous changes for weights
        self.d_w_prev = [np.zeros_like(self.w[i]) for i in range(self.L-1)]

        # Thresholds
        self.theta = [np.random.randn(1, layers[i+1]) * 0.01 for i in range(self.L-1)]
        # Xavier intialization
        # self.theta = [np.random.randn(n, 1) * np.sqrt(2 / (layers[i])) for i, n in enumerate(layers[1:])]
        # He initialization
        # self.theta = [np.random.randn(layers[i], layers[i+1]) * np.sqrt(2. / layers[i]) for i in range(self.L - 1)]
        
        # Thresholds changes
        self.d_theta = [np.zeros_like(self.theta[i]) for i in range(self.L-1)]
        # Previous changes for thresholds
        self.d_theta_prev = [np.zeros_like(self.theta[i]) for i in range(self.L-1)]

        # Propagation of error
        self.delta = [np.zeros((1, layers[i])) for i in range(self.L)]

        # Loss epochs storage
        self.train_loss = np.zeros(self.epochs) 

    ##############################
    # PRIVATE METHODS DEFINITION #
    ##############################

    def _select_activation_function(self, x):
        if self.fact == 'sigmoid':
            return sigmoid(x)
        elif self.fact == 'relu':
            return relu(x)
        elif self.fact == 'leaky_relu':
            return leaky_relu(x)
        elif self.fact == 'tanh':
            return tanh(x)
        elif self.fact == 'linear':
            return linear(x)
        else:
            return leaky_relu(x)
    
    def _select_derivative_activation_function(self, x):
        if self.fact == 'sigmoid':
            return sigmoid_derivative(x)
        elif self.fact == 'relu':
            return relu_derivative(x)
        elif self.fact == 'leaky_relu':
            return leaky_relu_derivative(x)
        elif self.fact == 'tanh':
            return tanh_derivative(x)
        elif self.fact == 'linear':
            return linear_derivative(x)
        else:
            return leaky_relu_derivative(x)

    def _forward_propagation(self, X):
        # Get the input data
        self.xi[0] = X
        # Activate the forward propagation for each layer
        for i in range(1, self.L):
            # Perform a product between the weights of the previous layer and the activations of the previous layer, and then adds the theta
            self.h[i] = np.dot(self.xi[i-1], self.w[i-1]) + self.theta[i-1]
            # Get the activation of the processed layer
            self.xi[i] = self._select_activation_function(self.h[i])
        # Return the last layer activation value that corresponds to the output layer
        return self.xi[-1]
    
    def _backward_propagation(self, X, y, output):
        # Compute the error at the output layer
        output_error = y - output
        # Compute the gradient of the loss with respect to the output
        self.delta[-1] = output_error * self._select_derivative_activation_function(self.h[-1])

        # Backpropagate the error through each layer
        for i in range(self.L-2, -1, -1):
            self.delta[i] = self.delta[i+1].dot(self.w[i].T) * self._select_derivative_activation_function(self.h[i])
            # Gradient clipping
            np.clip(self.delta[i], -1, 1, out=self.delta[i])
       
        # Update the weights and threshold using gradient descent with momentum
        for i in range(self.L-1):
            # Gradient of the weights
            self.d_w[i] = self.xi[i].T.dot(self.delta[i+1])  
            # Gradient of the threshold
            self.d_theta[i] = np.sum(self.delta[i+1], axis=0, keepdims=True)  

            # Gradient clipping
            np.clip(self.d_w[i], -1, 1, out=self.d_w[i])
            np.clip(self.d_theta[i], -1, 1, out=self.d_theta[i])

            # Apply momentum and update weights and biases
            self.w[i] += self.learning_rate * self.d_w[i] + self.momentum * self.d_w_prev[i]
            self.theta[i] += self.learning_rate * self.d_theta[i] + self.momentum * self.d_theta_prev[i]

            # Save the previous weight and bias updates for momentum
            self.d_w_prev[i] = self.d_w[i]
            self.d_theta_prev[i] = self.d_theta[i]

    def _train(self, X, y, batch_size=32):
        # Normalize input data
        X = (X - X.min()) / (X.max() - X.min())
        
        for epoch in range(self.epochs):
            # Mini-batch gradient descent
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                output = self._forward_propagation(X_batch)
                self._backward_propagation(X_batch, y_batch, output)
            # Calculate loss of each epoch
            output = self._forward_propagation(X)
            self._backward_propagation(X, y, output)
            loss = np.mean(np.square(y - output))
            # Store loss of each epoch during the training
            self.train_loss[epoch] = loss
            # Print the loss every 1000 epochs for debug purpose
            # if epoch % 1000 == 0:  
            #     print(f"Epoch {epoch}, Loss: {loss}")

    #############################
    # PUBLIC METHODS DEFINITION #
    #############################

    def fit(self, X, y):
        # Normalize input data
        X = (X - X.min()) / (X.max() - X.min())
        # Train system
        self._train(X, y)

    def predict(self, X):
        # Normalize input data
        X = (X - X.min()) / (X.max() - X.min())
        # Return prediction
        return self._forward_propagation(X)
    
    def loss_epochs(self):
        return self.train_loss