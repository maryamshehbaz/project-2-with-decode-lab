# Project 2: Data Classification Using AI
# Name: Rule-Based AI Project 2
# Algorithm: K-Nearest Neighbors (KNN)

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = load_iris()

# Create a DataFrame
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data["Species"] = iris.target

print("First 5 Rows of Dataset:")
print(data.head())

# Features (X) and Target (y)
X = data.iloc[:, :-1]
y = data["Species"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create KNN classifier
model = KNeighborsClassifier(n_neighbors=3)

# Train the model
model.fit(X_train, y_train)

# Predict test data
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# Display predictions
print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(predictions)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, predictions))