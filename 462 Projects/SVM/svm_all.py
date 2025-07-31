import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from cvxopt import matrix, solvers
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import time
import warnings
warnings.filterwarnings('ignore')

def print_model_analysis(model_name, metrics, test_precision, test_recall, test_f1, conf_matrix):
    print(f"\n{model_name} Results:")
    print("-" * (len(model_name) + 9))
    
    print("\nMetrics:")
    for phase in ['Training', 'Testing']:
        print(f"\n{phase}:")
        for metric, value in metrics[phase].items():
            print(f"{metric}: {value:.4f}")
    
    print(f"\nTest Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    
    print("\nConfusion Matrix:")
    print(pd.DataFrame(conf_matrix))

def get_analysis( y_test, y_test_pred, y_train, y_train_pred, test_prob, train_prob):
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average='weighted'
        )
    metrics = {
    'Training': {
        'Accuracy': accuracy_score(y_train, y_train_pred),
        'ROC AUC': roc_auc_score(y_train, train_prob, multi_class='ovr'),
    },
    'Testing': {
        'Accuracy': accuracy_score(y_test, y_test_pred),
        'ROC AUC': roc_auc_score(y_test, test_prob, multi_class='ovr'),
    }
    }
    conf_matrix = confusion_matrix(y_test, y_test_pred)

    return metrics, test_precision, test_recall, test_f1, conf_matrix

class LinearQPSVM():
    def __init__(self, C=1.0):
        self.C = C
        self.w = None
        self.b = None
        
    def get_params(self, deep=True):
        return {"C": self.C}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Construct the quadratic programming matrices
        P = matrix(np.dot(X, X.T) * np.outer(y, y))
        q = matrix(-np.ones(n_samples))
        
        # Inequality constraints
        G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
        h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))
        
        # Equality constraint
        A = matrix(y.reshape(1, -1).astype(float))
        b = matrix(np.zeros(1))
        
        # Solve the quadratic programming problem
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(solution['x']).flatten()
        
        # Find support vectors
        sv_indices = alphas > 1e-7 #a small number to filter out non-support vectors
        self.alphas = alphas[sv_indices]
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]
        
        # Calculate weights and bias using the formula w = sum(alpha_i * y_i * x_i)
        self.w = np.sum(self.alphas.reshape(-1, 1) * 
                       self.support_vector_labels.reshape(-1, 1) * 
                       self.support_vectors, axis=0)
        # Calculate bias using the formula b = mean(y_i - w^T * x_i) or just w·x + b = 0,
        
        margins = np.dot(self.support_vectors, self.w)
        self.b = np.mean(self.support_vector_labels - margins)
        
        return self
    
    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
    
    def decision_function(self, X):
        return np.dot(X, self.w) + self.b
    def predict_proba(self, X):
        # Convert decision function values to probabilities using softmax
        scores = self.decision_function(X)
        # Scale scores to avoid numerical issues
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)
        # Return probabilities for both classes
        return np.vstack([1 - probs, probs]).T

class MulticlassLinearQPSVM():
    def __init__(self, C=1.0):
        self.C = C
        self.classifiers = {}
        
    def get_params(self, deep=True):
        return {"C": self.C}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        
    def fit(self, X, y):
        self.classes = np.unique(y)#now this value is 3 but at first we were thinking about making it 5
        n_classes = len(self.classes)
        
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                mask = np.logical_or(y == self.classes[i], y == self.classes[j])
                X_binary = X[mask]
                y_binary = y[mask]
                y_binary = np.where(y_binary == self.classes[i], -1, 1)
                
                classifier = LinearQPSVM(C=self.C)
                classifier.fit(X_binary, y_binary)
                self.classifiers[(i, j)] = classifier
        
        return self
    
    def predict(self, X):
        n_samples = X.shape[0]
        votes = np.zeros((n_samples, len(self.classes)))
        
        for (i, j), classifier in self.classifiers.items():
            predictions = classifier.predict(X)
            votes[predictions < 0, i] += 1
            votes[predictions > 0, j] += 1
        
        return self.classes[votes.argmax(axis=1)]

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
    
    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        probs = np.zeros((n_samples, n_classes))
        
        # Get decision scores from each binary classifier
        for (i, j), classifier in self.classifiers.items():
            binary_probs = classifier.predict_proba(X)
            probs[:, i] += binary_probs[:, 0]  # probability for class i
            probs[:, j] += binary_probs[:, 1]  # probability for class j
            
        # Normalize probabilities
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

# Load and prepare data
print("Loading and preparing data...")
data = pd.read_csv("./prepare_dataset/data/dataset_new_cat.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 1. Custom Implementation
print("\n1. Custom Implementation Results")
basic_model = MulticlassLinearQPSVM(C=1.0)
start_time = time.time()
basic_model.fit(X_train, y_train)

# Get predictions
y_train_pred = basic_model.predict(X_train)
y_test_pred = basic_model.predict(X_test)


# Get probability predictions
train_prob = basic_model.predict_proba(X_train)
test_prob = basic_model.predict_proba(X_test)

# Get analysis
metrics, test_precision, test_recall, test_f1, conf_matrix = get_analysis(
    y_test, y_test_pred, y_train, y_train_pred, test_prob, train_prob
)

train_time = time.time() - start_time
print(f"\nTraining time: {train_time:.2f} seconds")
print_model_analysis("Custom Linear QP SVM", metrics, test_precision, test_recall, test_f1, conf_matrix)

# 2. Scikit-learn Implementation
kernel_configs = {
    'linear': {
        'kernel': ['linear'],
        'C': [0.1, 1.0, 10.0, 100.0]
    },
    'polynomial': {
        'kernel': ['poly'],
        'C': [0.1, 1.0, 10.0, 100.0],
        'degree': [2, 3],
        'gamma': ['scale']
    },
    'rbf': {
        'kernel': ['rbf'],
        'C': [0.1, 1.0, 10.0, 100.0],
        'gamma': ['scale', 'auto']
    }
}

for kernel_name, param_grid in kernel_configs.items():
    print(f"\nTraining {kernel_name} kernel SVM")
    start_time = time.time()
    
    # Create and train model with probability estimation enabled
    base_svm = SVC(probability=True)
    grid_search = GridSearchCV(base_svm, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Get predictions and probabilities
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    train_prob = best_model.predict_proba(X_train)
    test_prob = best_model.predict_proba(X_test)
    
    # Get analysis
    metrics, test_precision, test_recall, test_f1, conf_matrix = get_analysis(
        y_test, y_test_pred, y_train, y_train_pred, test_prob, train_prob
    )
    
    train_time = time.time() - start_time
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Training time: {train_time:.2f} seconds")
    print_model_analysis(f"Scikit-learn {kernel_name.upper()} SVM", metrics, test_precision, test_recall, test_f1, conf_matrix)