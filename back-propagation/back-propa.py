import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Data
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
              [8764, 10222, 10002, 8905, 15023]])

# Reshape data to represent months over the years
X_reshaped = X.reshape(-1)

# Create sliding window input-output pairs
window_size = 12
X_data = []
y_data = []
for i in range(len(X_reshaped) - window_size):
    X_data.append(X_reshaped[i:i + window_size])
    y_data.append(X_reshaped[i + window_size])

X_data = np.array(X_data)
y_data = np.array(y_data)

# Data split
X_train = X_data[:24]  # Data 2016-2018 (24 samples of 12 months each)
y_train = y_data[:24]  # Target 2017-2019 (24 months)
X_test = X_data[24:36]  # Data 2019-2020 (12 samples of 12 months each)
y_test = y_data[24:36]  # Target 2020 (12 months)

# Normalize data
X_train = X_train / np.max(X_reshaped)
y_train = y_train / np.max(X_reshaped)
X_test = X_test / np.max(X_reshaped)
y_test = y_test / np.max(X_reshaped)

# Build the model
model = Sequential()
model.add(Dense(10, input_dim=window_size, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=100, verbose=1)

# Test the model
predictions = model.predict(X_test)

# Denormalize predictions
predictions = predictions * np.max(X_reshaped)

print("Prediksi jumlah penerima bantuan sembako tahun 2021:")
print(predictions)

# Total prediksi penerima bantuan sembako tahun 2021
total_prediksi_2021 = np.sum(predictions)
print("Total prediksi penerima bantuan sembako tahun 2021:", total_prediksi_2021)
