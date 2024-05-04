import numpy as np

# Fungsi aktivasi sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Fungsi turunan sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Data latih dan target
X = np.array([[0.3997, 0.384, 0.3877],
              [0.3865, 0.1806, 0.3855],
              [0.2754, 0.2503, 0.1815],
              [0.2773, 0.3876, 0.2904],
              [0.3004, 0.3229, 0.275],
              [0.2763, 0.387, 0.2256],
              [0.0775, 0.1894, 0.162],
              [0.2996, 0.2775, 0.3877],
              [0.166, 0.51, 0.4098],
              [0.3066, 0.4017, 0.3975],
              [0.2898, 0.3545, 0.3985],
              [0.2762, 0.4209, 0.399]])

Y = np.array([[0.1905],
              [0.2996],
              [0.1896],
              [0.1],
              [0.199],
              [0.2659],
              [0.3866],
              [0.4179],
              [0.3967],
              [0.2754],
              [0.1662],
              [0.2902]])

# Inisialisasi bobot dan bias secara acak
np.random.seed(1)
input_neurons = 3
hidden_neurons = 5
output_neurons = 1

weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))
bias_hidden = np.random.uniform(size=(1, hidden_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))

# Hyperparameters
learning_rate = 0.2
epochs = 10000
target_error = 0.001

# Training the neural network using Backpropagation
for epoch in range(epochs):
    # Forward pass
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # Backpropagation
    error = Y - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Update weights and bias
    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    # Calculate mean squared error
    mean_squared_error = np.mean(np.square(error))

    # Early stopping if target error is reached
    if mean_squared_error <= target_error:
        print(f"Training converged at epoch {epoch+1}")
        break

# Uji data pada tahun 2021
data_uji = np.array([[0.897, 0.9093, 0.8956]])  # Data pada bulan pertama 2021

hidden_layer_input = np.dot(data_uji, weights_input_hidden) + bias_hidden
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
predicted_output = sigmoid(output_layer_input)

predicted_amount = predicted_output * 112498  # Scaling hasil prediksi ke jumlah penerima bantuan
print(f"Prediksi jumlah penerima bantuan sembako pada tahun 2021: {predicted_amount[0][0]} Kepala Keluarga")
