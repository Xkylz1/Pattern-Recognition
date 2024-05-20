import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image
import numpy as np

# Load the dataset
df = pd.read_csv('C:/Users/ACER/Documents/Pattern-Recognition/decision-tree/lung_cancer_examples.csv')

# Display the first few rows of the dataset
df_head = df.head()
print(df_head)

# Preprocess Data
# Select columns for features and target
X = df.iloc[:, 2:-1].values
y = df.iloc[:, -1].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and Train Gradient Boosting Machine Model
gbm = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbm.fit(X_train, y_train)

# Make Predictions and Evaluate
y_pred = gbm.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Print detailed classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Test with a Specific Sample
sample = np.array([[30, 0, 5, 2]])  # Replace with your actual sample
prediction = gbm.predict(sample)

print('Prediction for sample [30, 0, 5, 2]:', prediction[0])

# Extract one tree from the forest (e.g., the first tree)
tree = gbm.estimators_[0, 0]  # For GBM, access the tree at [0, 0] index

# Export the tree as DOT data with simplified visualization parameters
dot_data = export_graphviz(tree, out_file=None, 
                           feature_names=df.columns[2:-1],
                           class_names=[str(i) for i in set(y)],
                           filled=True, rounded=True,
                           special_characters=True,
                           max_depth=3,  # Limit the depth for better readability
                           node_ids=False,  # Do not show node IDs
                           label='none',  # Do not show labels
                           leaves_parallel=False)  # Avoid parallel leaves

# Use pydotplus to create a graph from the DOT data
graph = pydotplus.graph_from_dot_data(dot_data)

# Display the simplified tree image
Image(graph.create_png(), width=800, height=600)