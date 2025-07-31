from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv("dataset_new_cat.csv")

# Balance the dataset using undersampling (Reduce the majority class)
class_counts = data['density'].value_counts()
print("Class distribution before balancing:")
print(class_counts)

min_samples = class_counts.min()
balanced_data = pd.DataFrame()

# Perform undersampling by reducing the majority class
for class_label in class_counts.index:
    class_data = data[data['density'] == class_label]
    if len(class_data) > min_samples:
        class_data_undersampled = resample(
            class_data, 
            replace=False,  # Undersample without replacement
            n_samples=min_samples,  # Match the minimum class count
            random_state=1234
        )
    else:
        class_data_undersampled = class_data
    balanced_data = pd.concat([balanced_data, class_data_undersampled])

# Print class distribution after balancing using undersampling
print("Class distribution after balancing using undersampling:")
print(balanced_data['density'].value_counts())

# Separate features (X) and target (y)
X = balanced_data.drop(columns=['density']).iloc[:, :-1].values  # Features
y = balanced_data['density'].values  # Target

# Split the resampled data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Feature Scaling (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define class weights (penalize class 1 more than class 2 and class 2 more than class 0)
class_weights = {0: 1, 1: 3, 2: 2}  # Example weights: 1 for class 0, 2 for class 1, and 3 for class 2

# Define Logistic Regression model with class weights
model = LogisticRegression(max_iter=50000, class_weight=class_weights, random_state=1234)

# Optionally, you can use GridSearchCV to tune hyperparameters
param_grid = {
    'C': [0.0001,0.001, 0.01, 1, 10],
    'solver': ['liblinear', 'saga']
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train_scaled, y_train)

# Print the best parameters found by GridSearchCV
print("Best Parameters from Grid Search:", grid_search.best_params_)

# Train the model using the best parameters
best_model = grid_search.best_estimator_
best_model.fit(X_train_scaled, y_train)

# Make predictions
predictions = best_model.predict(X_test_scaled)

# Calculate overall accuracy
overall_accuracy = np.sum(y_test == predictions) / len(y_test)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)

# Generate classification report
class_report = classification_report(y_test, predictions, target_names=["Class 0", "Class 1", "Class 2"])

# Calculate ROC-AUC
roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test_scaled), multi_class='ovr')

# Print the detailed report
print("Overall Accuracy:", overall_accuracy)
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
print("\nROC-AUC Score:", roc_auc)
