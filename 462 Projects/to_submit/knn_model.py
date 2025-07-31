import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_score, recall_score, f1_score
import time
from collections import Counter

def euclidean_distance(x1, x2):
    # return euclidean distance between 2 vectors
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
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
    def predict_proba(self, X):
            return [self._predict_proba(x) for x in X]

    def _predict_proba(self, x):
        # Compute distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Get the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Count labels
        label_counts = Counter(k_nearest_labels)
        
        # Compute probabilities
        total = self.k
        probabilities = {label: count / total for label, count in label_counts.items()}
        
        # Ensure all classes are present in probabilities
        all_classes = np.unique(self.y_train)
        proba_array = np.array([probabilities.get(cls, 0) for cls in all_classes])
        
        return proba_array


# Load your dataset
data = pd.read_csv("dataset_3classes_3_cat.csv")

# Separate features and target
X = data.iloc[:, :-1].values  # Exclude 'density' (last column)
y = data.iloc[:, -1].values   # Use 'density' as the target

# Cross-validation setup (5-fold)
kf = KFold(n_splits=5, shuffle=True, random_state=1234)

best_k = None
best_accuracy = 0
accuracies = []

for k in range(3, 12, 2):
    clf = KNN(k=k)
    fold_accuracies = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        # Compute accuracy
        acc = np.sum(predictions == y_test) / len(y_test)
        fold_accuracies.append(acc)

    # Average accuracy for the current k
    avg_accuracy = np.mean(fold_accuracies)
    accuracies.append(avg_accuracy)
    print(f"k: {k} with Accuracy: {avg_accuracy}")
    # If this is the best accuracy, store the value of k
    if avg_accuracy > best_accuracy:
        best_accuracy = avg_accuracy
        best_k = k




# Train with the best k
clf = KNN(k=best_k)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

start_train = time.time()
clf.fit(X_train, y_train)
end_train = time.time()
y_pred_train = clf.predict(X_train)

# Evaluate on the whole dataset for training evaluation
acc_train = np.sum(y_pred_train == y_train) / len(y)
roc_auc_train = roc_auc_score(y_train, clf.predict_proba(X_train),multi_class='ovr')
precision_train = precision_score(y_train, y_pred_train, average='macro')
recall_train = recall_score(y_train, y_pred_train, average='macro')
f1_train = f1_score(y_train, y_pred_train, average='macro')

start_test = time.time()
y_pred_test = clf.predict(X_test)
end_test = time.time()

# Evaluate for testing set
acc_test = np.sum(y_pred_test == y_test) / len(y_test)
roc_auc_test = roc_auc_score(y_test, clf.predict_proba(X_test),multi_class='ovr')
precision_test = precision_score(y_test, y_pred_test, average='macro')
recall_test = recall_score(y_test, y_pred_test, average='macro')
f1_test = f1_score(y_test, y_pred_test, average='macro')

# Print results
print(f"Best k: {best_k} with Accuracy: {best_accuracy}")
print("\nTraining Metrics:")
print(f"Accuracy: {acc_train}")
print(f"ROC AUC: {roc_auc_train}")
print(f"Precision: {precision_train}")
print(f"Recall: {recall_train}")
print(f"F1-Score: {f1_train}")
print(f"Training Time: {end_train - start_train} seconds")

print("\nTesting Metrics:")
print(f"Accuracy: {acc_test}")
print(f"ROC AUC: {roc_auc_test}")
print(f"Precision: {precision_test}")
print(f"Recall: {recall_test}")
print(f"F1-Score: {f1_test}")
print(f"Testing Time: {end_test - start_test} seconds")

print(f"Confusion matrix: {confusion_matrix(y_test, y_pred_test)}")
print(f"Classification report: {classification_report(y_test, y_pred_test)}")

