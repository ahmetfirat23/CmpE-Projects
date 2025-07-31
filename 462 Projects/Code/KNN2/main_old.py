import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    # return ecludian distance between 2 vectors
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # compute the distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
    
        # get the closest k
        k_indices = np.argsort(distances)[:self.k] 
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]

# Load your dataset
data = pd.read_csv("important_data.csv")

# Separate features and target
X = data.iloc[:, :-1].values  # Exclude  'density' (last column)
y = data.iloc[:, -1].values    # Use 'density' as the target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Create a file to write the results
with open('results_important_5_3.txt', 'w') as file:


    k=5
    # Train the KNN classifier
    clf = KNN(k=k)
    clf.fit(X_train, y_train)

    # Make predictions
    predictions = clf.predict(X_test)
    file.write(f"k = {k}")
    # Write detailed prediction results to the file
    # file.write("Predictions vs Actual values:\n")
    # for pred, actual in zip(predictions, y_test):
    #     file.write(f"Predicted: {pred}, Actual: {actual}\n")

    # Confusion Matrix
    file.write("\nConfusion Matrix:\n")
    cm = confusion_matrix(y_test, predictions)
    file.write(np.array2string(cm) + '\n')

    # Classification Report (Precision, Recall, F1-score)
    file.write("\nClassification Report:\n")
    report = classification_report(y_test, predictions)
    file.write(report + '\n')

    # Accuracy of the model
    acc = np.sum(predictions == y_test) / len(y_test)
    file.write("\nAccuracy: " + str(acc) + '\n')

    # Additional: Accuracy per Class (class-wise accuracy)
    classes = np.unique(y)
    for cls in classes:
        cls_accuracy = np.sum((predictions == cls) & (y_test == cls)) / np.sum(y_test == cls)
        file.write(f"Accuracy for class {cls}: {cls_accuracy:.2f}\n")

print("Results have been written to 'results.txt'.")
