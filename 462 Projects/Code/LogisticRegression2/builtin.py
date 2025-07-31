import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load the dataset
data = pd.read_csv("dataset_3classes_3_cat.csv")

# Separate features (X) and target (y)
X = data.iloc[:, :-1].values  #  'density' (last column)
y = data.iloc[:, -1].values    # Use 'density' as the target

# # Scale the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Initialize the Logistic Regression classifier with L2 regularization
# 'C' controls regularization strength: smaller 'C' means stronger regularization
logreg = LogisticRegression(max_iter=1000000, solver='lbfgs', C=0.9) 
# logreg = LogisticRegression(max_iter=1000000, solver='saga', C=0.1) 

# Train the model
logreg.fit(X_train, y_train)

# Make predictions
predictions = logreg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
class_report = classification_report(y_test, predictions)

# Calculate training accuracy
train_predictions = logreg.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)


# Print the results to the console
print(f"Test Accuracy: {accuracy}")
print(f"Training Accuracy: {train_accuracy}\n")

print("Confusion Matrix:")
print(np.array2string(conf_matrix) + '\n')

print("Classification Report:")
print(class_report)

print("Logistic Regression model results with regularization have been printed to the console.")
