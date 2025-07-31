import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

# Load the dataset
data = pd.read_csv("dataset_3classes_3_cat.csv")

# Separate features (X) and target (y)
X = data.iloc[:, :-1].values  # All columns except the target density
y = data.iloc[:, -1].values   # Last column as the target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# Define a parameter grid
param_grid = {
    'C': [0.1, 0.5, 0.9, 1, 10],  # Regularization strength
    'solver': ['saga', 'liblinear'],  # Optimization algorithms
    'class_weight': [None, {0: 1, 1: 2, 2: 2}],  # Class weights
    'max_iter': [10000, 50000, 100000]  # Maximum number of iterations
}

# Initialize the Logistic Regression model
logreg = LogisticRegression(random_state=1234)

# Perform GridSearchCV
grid_search = GridSearchCV(
    estimator=logreg, 
    param_grid=param_grid, 
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',  # Metric to optimize
)

# Measure training runtime
start_training_time = time.time()
grid_search.fit(X_train, y_train)
training_runtime = time.time() - start_training_time

# Best parameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
print("Best Parameters:", best_params)

# Predictions for testing set
start_testing_time = time.time()
predictions = best_model.predict(X_test)
testing_runtime = time.time() - start_testing_time

# Compute metrics for the testing set
test_accuracy = accuracy_score(y_test, predictions)
test_precision = precision_score(y_test, predictions, average="weighted")
test_recall = recall_score(y_test, predictions, average="weighted")
test_f1 = f1_score(y_test, predictions, average="weighted")
test_roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test), multi_class="ovo", average="weighted")

# Compute metrics for the training set
train_predictions = best_model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)
train_precision = precision_score(y_train, train_predictions, average="weighted")
train_recall = recall_score(y_train, train_predictions, average="weighted")
train_f1 = f1_score(y_train, train_predictions, average="weighted")
train_roc_auc = roc_auc_score(y_train, best_model.predict_proba(X_train), multi_class="ovo", average="weighted")

# Print training results
print("Training Results:")
print(f"Accuracy: {train_accuracy:.4f}")
print(f"ROC AUC: {train_roc_auc:.4f}")
print(f"Precision: {train_precision:.4f}")
print(f"Recall: {train_recall:.4f}")
print(f"F1-score: {train_f1:.4f}")
print(f"Training Runtime: {training_runtime:.4f} seconds\n")

# Print testing results
print("Testing Results:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"ROC AUC: {test_roc_auc:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1-score: {test_f1:.4f}")
print(f"Testing Runtime: {testing_runtime:.4f} seconds\n")

# Print confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, predictions)
class_report = classification_report(y_test, predictions)

print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
