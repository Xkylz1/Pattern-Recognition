# Kelompok 1
# Nama Anggota
# 1. NANDA PUTRI RAHMAWATI (2011016320021)
# 2. HELMA MUKIMAH (2211016220008)
# 3. NORKHADIJAH (2211016220030)
# 4. FAUZAN SAPUTRA (2211016310003)
# Link GDrive data dan output = https://drive.google.com/drive/folders/1f9xxjJve0hVL2KrljdZDwvG4Gr91DkdY?usp=sharing

import numpy as np

# Fungsi aktivasi sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Fungsi turunan sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Data latih dan target
X = np.array([[10009, 9850, 9888, 7899, 9005],
              [9876, 7800, 9866, 8999, 8990],
              [8755, 8502, 7809, 7890, 12067],
              [8775, 9887, 8907, 6987, 14012],
              [9008, 9234, 8751, 7985, 11002],
              [8765, 9881, 8253, 8660, 14004],
              [6760, 7888, 7612, 9877, 14034],
              [8999, 8777, 9888, 10192, 15006],
              [7652, 11121, 10111, 9978, 15053],
              [9070, 10029, 9987, 8755, 14889],
              [8901, 9553, 9997, 7654, 13988],
              [8764, 10222, 10002, 8905, 15023]]) #data latih 2016-2020

Y = np.array([[7899],
              [8990],
              [7890],
              [6987],
              [7985],
              [8660],
              [9877],
              [10192],
              [9978],
              [8755],
              [7654],
              [8905]]) #data target latih 2019

# Normalisasi data
a = np.max(X)
b = np.min(X)
X_normalized = 0.8 * (X - b) / (a - b) + 0.1
Y_normalized = 0.8 * (Y - b) / (a - b) + 0.1

# Inisialisasi bobot dengan rentang -0.5 sampai 0.5
np.random.seed(1)
weights_input_hidden = np.random.uniform(low=-0.5, high=0.5, size=(5, 5))
weights_hidden_output = np.random.uniform(low=-0.5, high=0.5, size=(5, 1))
bias_hidden = np.random.uniform(size=(1, 5))
bias_output = np.random.uniform(size=(1, 1))

# Hyperparameters
learning_rate = 0.2
epochs = 10000
target_error = 0.001

# Training the neural network using Backpropagation
for epoch in range(epochs):
    # Forward pass
    hidden_layer_input = np.dot(X_normalized, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # Backpropagation
    error = Y_normalized - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Update weights and bias
    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    weights_input_hidden += X_normalized.T.dot(d_hidden_layer) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    # Calculate mean squared error
    mean_squared_error = np.mean(np.square(error))

    # Early stopping if target error is reached
    if mean_squared_error <= target_error:
        print(f"Training converged at epoch {epoch+1}")
        break

# Data uji 2020 + (random value)
data_uji = np.array([[7899, 8990, 7890, 6987, 9005],  # Januari
                     [9850, 7800, 8502, 9887, 8990],  # Februari
                     [9888, 9866, 7809, 8907, 12067],  # Maret
                     [7899, 8999, 7890, 6987, 14012],  # April
                     [9005, 8990, 12067, 14012, 11002],  # Mei
                     [8765, 9881, 8253, 8660, 14004],  # Juni
                     [6760, 7888, 7612, 9877, 14034],  # Juli
                     [8999, 8777, 9888, 10192, 15006],  # Agustus
                     [7652, 11121, 10111, 9978, 15053],  # September
                     [9070, 10029, 9987, 8755, 14889],  # Oktober
                     [8901, 9553, 9997, 7654, 13988],  # November
                     [8764, 10222, 10002, 8905, 15023]])  # Desember


data_uji_normalized = 0.8 * (data_uji - b) / (a - b) + 0.1

hidden_layer_input = np.dot(data_uji_normalized, weights_input_hidden) + bias_hidden
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
predicted_output = sigmoid(output_layer_input)

# Denormalisasi hasil prediksi
predicted_amount = predicted_output * (a - b) / 0.8 + b

# Jumlahkan prediksi untuk seluruh tahun 2021
total_predicted_amount = np.sum(predicted_amount)
print(f"Prediksi jumlah penerima bantuan sembako pada tahun 2021: {total_predicted_amount} Kepala Keluarga")