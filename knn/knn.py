# Kelompok 1
# Nama Anggota
# 1. NANDA PUTRI RAHMAWATI (2011016320021)
# 2. HELMA MUKIMAH (2211016220008)
# 3. NORKHADIJAH (2211016220030)
# 4. FAUZAN SAPUTRA (2211016310003)
# Link GDrive data dan output = https://drive.google.com/drive/folders/1-dgVW5mK2UjWzQTQIz2j16YxZ6oczWcy?usp=drive_link

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Load the dataset
S = pd.read_csv('C:/Users/ACER/Documents/Pattern-Recognition/knn/lung_cancer_examples.csv')
S = S.values

# Separate input features and labels
X = S[:, 2:6]  # Input features (Age, Smokes, AreaQ, Alkhol)
y = S[:, 6]    # Labels (Result)

# Ensure that the labels are integers
y = y.astype(int)

# Initialize K-Fold cross-validation
kf = KFold(n_splits=5, random_state=0, shuffle=True)

# Print KFold object to understand its structure
print(kf)

# Loop over different values of k
results = {}  # Dictionary to store results for each k

for k in range(1, 6):
    neigh2 = KNeighborsClassifier(n_neighbors=k, metric='euclidean')

    # Initialize arrays to store accuracy, precision, and recall for each fold
    avg_acc = np.zeros(5)
    avg_pre = np.zeros(5)
    avg_rec = np.zeros(5)

    # Cross-validation loop
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train the classifier
        neigh2.fit(X_train, y_train)

        # Make predictions
        y_pred = neigh2.predict(X_test)

        # Calculate and store metrics
        avg_acc[i] = accuracy_score(y_test, y_pred)
        avg_pre[i] = precision_score(y_test, y_pred, average='macro')
        avg_rec[i] = recall_score(y_test, y_pred, average='macro')

        # Print classification report for the current fold
        print(f"k = {k}, Fold {i+1} classification report:\n", classification_report(y_test, y_pred))

    # Store the average metrics for the current value of k
    results[k] = {
        "average_accuracy": np.mean(avg_acc),
        "average_precision": np.mean(avg_pre),
        "average_recall": np.mean(avg_rec)
    }

# Print results for each k and determine the best k based on average accuracy
best_k = 1
best_accuracy = 0

for k in results:
    print(f"Results for k = {k}:")
    print("Average Accuracy: ", results[k]["average_accuracy"])
    print("Average Precision: ", results[k]["average_precision"])
    print("Average Recall: ", results[k]["average_recall"])
    print()
    # Determine the best k based on the highest average accuracy
    if results[k]["average_accuracy"] > best_accuracy:
        best_accuracy = results[k]["average_accuracy"]
        best_k = k

print(f"The best k is {best_k} with an average accuracy of {best_accuracy}")
