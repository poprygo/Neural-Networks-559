import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(0)

# 1. Draw n = 300 real numbers uniformly at random on [0, 1]
x = np.random.uniform(0, 1, 300)

# 2. Draw n real numbers uniformly at random on [−1/10, 1/10]
nu = np.random.uniform(-0.1, 0.1, 300)

# 3. Let di = sin(20xi) + 3xi + νi
d = np.sin(20 * x) + 3 * x + nu

# Plot the points (xi, di) for steps 1-3
plt.scatter(x, d)
plt.xlabel('x')
plt.ylabel('d')
plt.title('Data Points (xi, di)')
plt.show()

# Neural Network Initialization
N = 24
weights_hidden = np.random.randn(1, N)
bias_hidden = np.random.randn(N)
weights_output = np.random.randn(N, 1)
bias_output = np.random.randn(1)
lr = 0.01

# Activation and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - np.tanh(x)**2

# Training Loop
num_epochs = 5000
mse_list = []

for epoch in range(num_epochs):
    total_error = 0
    for i in range(300):
        # Forward pass
        z_hidden = x[i] * weights_hidden + bias_hidden
        hidden_layer_output = tanh(z_hidden)
        prediction = np.dot(hidden_layer_output, weights_output) + bias_output
        
        # Compute error
        error = prediction - d[i]
        total_error += error**2
        
        # Backward pass
        d_output = 2 * error
        d_weights_output = hidden_layer_output * d_output
        d_bias_output = d_output
        
        d_hidden = d_output * weights_output.T * tanh_prime(z_hidden)
        d_weights_hidden = x[i] * d_hidden
        d_bias_hidden = d_hidden
        
        # Gradient Descent Update
        weights_output -= lr * d_weights_output.reshape(N, 1)
        bias_output -= lr * d_bias_output[0]  # Reshape the gradient here
        weights_hidden -= lr * d_weights_hidden
        bias_hidden -= lr * d_bias_hidden.reshape(N)
    
    mse_list.append((total_error/300)[0][0])

# Plot the number of epochs vs the MSE
plt.plot(mse_list)
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title('Epochs vs MSE')
plt.show()

# Plot the curve f(x, w0) as x ranges from 0 to 1 on top of the plot of points
x_test = np.linspace(0, 1, 300)
y_test = []

for xi in x_test:
    hidden_layer_output = tanh(xi * weights_hidden + bias_hidden)
    prediction = np.dot(hidden_layer_output, weights_output) + bias_output
    y_test.append(prediction[0])

plt.scatter(x, d, label='Data Points (xi, di)')
plt.plot(x_test, y_test, 'r-', label='NN Fit')
plt.xlabel('x')
plt.ylabel('d')
plt.legend()
plt.title('Data Points and Neural Network Fit')
plt.show()
