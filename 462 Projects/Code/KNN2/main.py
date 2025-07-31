import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import time
from knn_model import KNN
import os
import sys

# Delete __pycache__ directories in your project directory
def clear_python_cache():
    for root, dirs, files in os.walk("."):
        if "__pycache__" in dirs:
            cache_dir = os.path.join(root, "__pycache__")
            print(f"Deleting cache directory: {cache_dir}")
            for file in os.listdir(cache_dir):
                os.remove(os.path.join(cache_dir, file))
            os.rmdir(cache_dir)


def get_campus_accuracy(predictions, y_true, X_data):
    """
    Calculate accuracy for each campus.
    
    Args:
        predictions: Model predictions
        y_true: True labels
        X_data: Feature matrix where first column is campus
    
    Returns:
        dict: Dictionary with accuracy for each campus
    """
    campus_accuracies = {}
    unique_campuses = np.unique(X_data[:, 0])
    
    for campus in unique_campuses:
        # Convert boolean mask to indices
        campus_indices = np.where(X_data[:, 0] == campus)[0]
        if len(campus_indices) > 0:  # Only calculate if campus has data points
            campus_pred =[ predictions[i] for i in campus_indices]
            campus_true = [ y_true[i] for i in campus_indices]
            true_ones = sum([campus_pred[i] == campus_true[i] for i in range(len(campus_indices))])
            campus_accuracies[campus] = {
                'accuracy': true_ones/len(campus_indices),
                'samples': len(campus_indices)
            }
    
    return campus_accuracies

def cross_validate_knn(X, y, k_values, n_splits=5):
    """
    Perform k-fold cross-validation for multiple k values in KNN.
    
    Args:
        X: Features
        y: Target values
        k_values: List of k values to try
        n_splits: Number of folds for cross-validation
    
    Returns:
        dict: Average accuracy for each k value
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1234)
    k_scores = {k: [] for k in k_values}
    
    for k in k_values:
        for train_idx, val_idx in kf.split(X):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            clf = KNN(k=k)
            clf.fit(X_train_fold, y_train_fold)
            predictions = clf.predict(X_val_fold)
            accuracy = np.mean(predictions == y_val_fold)
            k_scores[k].append(accuracy)
    
    avg_scores = {k: np.mean(scores) for k, scores in k_scores.items()}
    return avg_scores

def evaluate_model(clf, X_train, X_test, y_train, y_test):
    """
    Evaluate the model on training and test sets.
    
    Returns:
        dict: Dictionary containing evaluation metrics and runtime
    """
    results = {}
    
    # Training time
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Prediction time
    start_time = time.time()
    train_predictions = clf.predict(X_train)
    test_predictions = clf.predict(X_test)
    predict_time = time.time() - start_time
    
    # Calculate metrics
    train_accuracy = np.mean(train_predictions == y_train)
    test_accuracy = np.mean(test_predictions == y_test)
    
    # Calculate campus-specific accuracies
    train_campus_accuracies = get_campus_accuracy(train_predictions, y_train, X_train)
    test_campus_accuracies = get_campus_accuracy(test_predictions, y_test, X_test)
    
    results['train_accuracy'] = train_accuracy
    results['test_accuracy'] = test_accuracy
    results['train_time'] = train_time
    results['predict_time'] = predict_time
    results['confusion_matrix'] = confusion_matrix(y_test, test_predictions)
    results['classification_report'] = classification_report(y_test, test_predictions)
    results['train_campus_accuracies'] = train_campus_accuracies
    results['test_campus_accuracies'] = test_campus_accuracies
    
    return results

# Main execution
if __name__ == "__main__":
    dataset="dataset_5classes_2_status"
    # Load and prepare data
    data = pd.read_csv(f"{dataset}.csv")
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    k_values = [3, 5, 7, 9, 11]
    cv_scores = cross_validate_knn(X_train, y_train, k_values)
    
    # Find optimal k
    optimal_k = max(cv_scores.items(), key=lambda x: x[1])[0]
    clear_python_cache() ### So Trained time is not grapped from caches
    # Train and evaluate final model with optimal k
    final_model = KNN(k=optimal_k)
    evaluation_results = evaluate_model(final_model, X_train, X_test, y_train, y_test)
    
    # Write results to file
    with open(f'{dataset}.txt', 'w') as file:
        # Cross-validation results
        file.write("Cross-validation Results:\n")
        for k, score in cv_scores.items():
            file.write(f"k={k}: {score:.4f}\n")
        file.write(f"\nOptimal k value: {optimal_k}\n")
        
        # Overall model performance
        file.write("\nOverall Model Performance:\n")
        file.write(f"Training Accuracy: {evaluation_results['train_accuracy']:.4f}\n")
        file.write(f"Test Accuracy: {evaluation_results['test_accuracy']:.4f}\n")
        campuses = {
            0: "Anadolu Hisari", 
            1: "Guney", 
            2: "Hisar", 
            3: "Kandilli", 
            4: "Kliyos", 
            5: "Kuzey"
        }
        # Campus-specific performance
        file.write("\nCampus-Specific Performance (Training Set):\n")
        for campus, metrics in evaluation_results['train_campus_accuracies'].items():
            file.write(f"Campus {int(campus)} {campuses[campus]}:\n")
            file.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
            file.write(f"  Number of samples: {metrics['samples']}\n")
        
        file.write("\nCampus-Specific Performance (Test Set):\n")
        for campus, metrics in evaluation_results['test_campus_accuracies'].items():
            file.write(f"Campus {int(campus)} {campuses[campus]}:\n")
            file.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
            file.write(f"  Number of samples: {metrics['samples']}\n")
        
        # Runtime
        file.write(f"\nTraining Time: {evaluation_results['train_time']:.4f} seconds\n")
        file.write(f"Prediction Time: {evaluation_results['predict_time']:.4f} seconds\n")
        
        
        file.write(f"\nClass 0 : Low, Class 1: Medium, Class 2: High\n")
        # Detailed metrics
        file.write("\nConfusion Matrix:\n")
        file.write(str(evaluation_results['confusion_matrix']))
        file.write("\n\nClassification Report:\n")
        file.write(evaluation_results['classification_report'])

    print("Results have been written to report.txt")