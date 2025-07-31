import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load the dataset
data = pd.read_csv("dataset_new_cat.csv")

# Separate features (X) and target (y)
X = data.iloc[:, :-1].values  #  'density' (last column)
y = data.iloc[:, -1].values    # Use 'density' as the target

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Initialize the Logistic Regression classifier with L2 regularization
# 'C' controls regularization strength: smaller 'C' means stronger regularization
logreg = LogisticRegression(max_iter=500, solver='lbfgs', C=0.1)  # Adjust C for regularization strength
# logreg = LogisticRegression(max_iter=500, solver='saga', penalty='l1', C=0.1)

# Train the model
logreg.fit(X_train, y_train)

# Make predictions
predictions = logreg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
class_report = classification_report(y_test, predictions)

# Create a file to write the results
with open('report_nonscaled.txt', 'w') as file:
    # Write the accuracy
    file.write(f"Accuracy: {accuracy}\n\n")

    # Write the confusion matrix
    file.write("Confusion Matrix:\n")
    file.write(np.array2string(conf_matrix) + '\n\n')

    # Write the classification report
    file.write("Classification Report:\n")
    file.write(class_report + '\n')



print("Logistic Regression model results with regularization have been written to 'logreg_results_with_regularization.txt'.")
