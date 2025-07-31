import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report
from knn_model import KNN

# Load your dataset
data = pd.read_csv("important_data.csv")

# Separate features and target
X = data.iloc[:, :-1].values  # Exclude 'density' (last column)
y = data.iloc[:, -1].values   # Use 'density' as the target

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Create a file to write the results
with open('results_important_5_3.txt', 'w') as file:
    # Perform 5-fold cross-validation to choose the best k value
    k_values = range(1, 21)  # Test k from 1 to 20
    best_k = None
    best_accuracy = 0
    kf = KFold(n_splits=5, shuffle=True, random_state=1234)

    for k in k_values:
        accuracies = []
        
        # Perform 5-fold cross-validation
        for train_index, val_index in kf.split(X_train):
            X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
            y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
            
            clf = KNN(k=k)
            clf.fit(X_fold_train, y_fold_train)
            predictions = clf.predict(X_fold_val)
            
            acc = np.sum(predictions == y_fold_val) / len(y_fold_val)
            accuracies.append(acc)
        
        # Calculate mean accuracy for this k value
        mean_accuracy = np.mean(accuracies)
        
        # Update best_k if this k performs better
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_k = k
    
    # Output the best k
    file.write(f"Best k (from 5-fold CV): {best_k}\n")

    # Train the KNN classifier with the best k
    clf = KNN(k=best_k)
    clf.fit(X_train, y_train)

    # Make predictions
    predictions = clf.predict(X_test)
    file.write(f"k = {best_k}\n")

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

print("Results have been written to 'results_important_5_3.txt'.")
