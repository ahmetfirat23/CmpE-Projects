from LogisticRegression import LogisticRegressionModel
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import pandas as pd

# Load the dataset
data = pd.read_csv("dataset_new_cat.csv")

# Balance the dataset
class_counts = data['density'].value_counts()
print("Class distribution before balancing:")
print(class_counts)

# Under sample
min_samples = class_counts.min()
balanced_data = pd.DataFrame()

for class_label in class_counts.index:
    class_data = data[data['density'] == class_label]
    class_data_balanced = resample(
        class_data, 
        replace=False,  # Undersample without replacement
        n_samples=min_samples,  # Match the minimum class count
        random_state=1234
    )
    balanced_data = pd.concat([balanced_data, class_data_balanced])

# Shuffle the balanced dataset
balanced_data = balanced_data.sample(frac=1, random_state=1234).reset_index(drop=True)
print("Class distribution after balancing:")
print(balanced_data['density'].value_counts())

# Separate features (X) and target (y)
X = balanced_data.iloc[:, :-1].values  # Features (all columns except the last one)
y = balanced_data.iloc[:, -1].values  # Target ('density')

# Initialize k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=1234)

# Define hyperparameter combinations
param_combinations = [
    {'lr': 0.0001, 'num_iter': 1000, 'reg_strength': 0.01},
    {'lr': 0.0001, 'num_iter': 5000, 'reg_strength': 0.01},
    {'lr': 0.0001, 'num_iter': 10000, 'reg_strength': 0.1},
    {'lr': 0.001, 'num_iter': 1000, 'reg_strength': 0.01},
    {'lr': 0.001, 'num_iter': 5000, 'reg_strength': 0.05},
    {'lr': 0.005, 'num_iter': 5000, 'reg_strength': 0.01},
    {'lr': 0.005, 'num_iter': 10000, 'reg_strength': 0.1},
    {'lr': 0.01, 'num_iter': 1000, 'reg_strength': 0.01}
]

# Open the file to write the detailed report
with open('detailed_report.txt', 'w') as report_file:
    # Write the header
    report_file.write("=== Detailed Report ===\n")
    
    for params in param_combinations:
        lr = params['lr']
        num_iter = params['num_iter']
        reg_strength = params['reg_strength']
        
        report_file.write(f"\n--- Hyperparameters: lr={lr}, num_iter={num_iter}, reg_strength={reg_strength} ---\n")
        
        # Variables to store results for the current combination
        all_accuracies = []
        all_conf_matrices = []
        all_class_reports = []

        # Perform 5-fold cross-validation
        for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
            report_file.write(f"\n--- Fold {fold + 1} ---\n")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train separate logistic regression models for each class
            regressor0 = LogisticRegressionModel(lr=lr, num_iter=num_iter, reg_strength=reg_strength)
            regressor0.fit(X_train, [1 if val == 0 else 0 for val in y_train])
            predictions0 = regressor0.predict(X_test)

            regressor1 = LogisticRegressionModel(lr=lr, num_iter=num_iter, reg_strength=reg_strength)
            regressor1.fit(X_train, [1 if val == 1 else 0 for val in y_train])
            predictions1 = regressor1.predict(X_test)

            regressor2 = LogisticRegressionModel(lr=lr, num_iter=num_iter, reg_strength=reg_strength)
            regressor2.fit(X_train, [1 if val == 2 else 0 for val in y_train])
            predictions2 = regressor2.predict(X_test)

            # Aggregate predictions and determine final predicted class
            result = [
                max(range(3), key=lambda i: [predictions0, predictions1, predictions2][i][index])
                for index in range(len(predictions0))
            ]

            # Calculate fold accuracy
            fold_accuracy = np.sum(y_test == result) / len(y_test)
            all_accuracies.append(fold_accuracy)

            # Generate confusion matrix
            conf_matrix = confusion_matrix(y_test, result)
            all_conf_matrices.append(conf_matrix)

            # Generate classification report
            class_report = classification_report(y_test, result, target_names=["Class 0", "Class 1", "Class 2"])
            all_class_reports.append(class_report)

            # Write results for the fold to the report
            report_file.write(f"Accuracy for Fold {fold + 1}: {fold_accuracy:.4f}\n")
            report_file.write("Confusion Matrix:\n")
            report_file.write(f"{conf_matrix}\n")
            report_file.write("Classification Report:\n")
            report_file.write(f"{class_report}\n")

        # Summary for the current parameter combination
        mean_accuracy = np.mean(all_accuracies)
        report_file.write(f"\n--- Summary for lr={lr}, num_iter={num_iter}, reg_strength={reg_strength} ---\n")
        report_file.write(f"Mean Accuracy: {mean_accuracy:.4f}\n")
        report_file.write(f"Accuracy per Fold: {all_accuracies}\n")
    
    print("Detailed report has been written to 'detailed_report.txt'.")
