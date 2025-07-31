import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pandas as pd
import time

# No Regularization
class SimpleLogisticRegressionModel:
    def __init__(self, lr=0.0001, num_iter=10000):
        self.lr=lr
        self.num_iter=num_iter
        self.weight=None
        self.bias=None

    def fit(self, X, Y):
        N , num_features= X.shape
        self.weight=np.zeros(num_features)
        self.bias=0
        for i in range(self.num_iter):
            linear_model= np.dot(X,self.weight)+self.bias # wX+b
            y_predicted=self._sigmoid_func(linear_model) # Sigmoid of (wX+b)
            # Gradient Descent
            # # dw, db comes from min of cross entropy loss function
            dw= (1/N) * np.dot(X.T, (y_predicted-Y)) 
            db=(1/N)* sum(y_predicted-Y)
            self.weight -= self.lr*dw # update weight 
            self.bias -= self.lr*db # updated bias
           
    def predict(self,X):
        linear_model= np.dot(X,self.weight)+self.bias # wX+b
        y_predicted=self._sigmoid_func(linear_model) # Sigmoid of (wX+b)
        return y_predicted
    def _sigmoid_func(self,x):
        return 1/(1+np.exp(-1*x))

class LogisticRegressionModel:
    def __init__(self, lr=0.0001, num_iter=1000000, reg_strength=0.1, early_stopping_rounds=5, tol=0.01e-4):
        self.lr = lr
        self.num_iter = num_iter
        self.weight = None
        self.bias = None
        self.reg_strength = reg_strength
        self.early_stopping_rounds = early_stopping_rounds
        self.tol = tol
        self.loss_history = []

    def _calculate_loss(self, X, Y, y_pred):
        """Calculate binary cross-entropy loss with L2 regularization"""
        N = X.shape[0]
        epsilon = 1e-15
        
        Y = np.array(Y).reshape(-1, 1)
        y_pred = np.array(y_pred).reshape(-1, 1)
        
        # Binary cross-entropy loss with vectorized operations
        loss = -np.mean(Y * np.log(y_pred + epsilon) + (1 - Y) * np.log(1 - y_pred + epsilon))
        
        # Add L2 regularization term
        l2_loss = (self.reg_strength / (2 * N)) * np.sum(np.square(self.weight))
        return float(loss + l2_loss)

    def fit(self, X, Y):
        N, num_features = X.shape
        self.weight = np.zeros((num_features, 1))
        self.bias = 0
        Y = np.array(Y).reshape(-1, 1)
        
        best_loss = float('inf')
        rounds_without_improvement = 0
        
        for i in range(self.num_iter):
            # Forward pass
            linear_model = np.dot(X, self.weight) + self.bias
            y_predicted = self._sigmoid_func(linear_model)
            
            # Calculate current loss
            current_loss = self._calculate_loss(X, Y, y_predicted)
            self.loss_history.append(current_loss)
            
            # Early stopping check
            if current_loss < best_loss - self.tol:
                best_loss = current_loss
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1
            
            # Stop if no improvement for several rounds
            if rounds_without_improvement >= self.early_stopping_rounds:
                print(f"Early stopping triggered at iteration {i+1}")
                break
                
            # Gradient Descent with L2 Regularization
            error = y_predicted - Y  # Shape: (N, 1)
            dw = (1/N) * np.dot(X.T, error) + (self.reg_strength / N) * self.weight  # Shape: (features, 1)
            db = (1/N) * np.sum(error)
            
            # Update parameters
            self.weight -= self.lr * dw
            self.bias -= self.lr * db
           
    def predict(self, X):
        linear_model = np.dot(X, self.weight) + self.bias
        y_predicted = self._sigmoid_func(linear_model)
        return y_predicted.reshape(-1)  

    def predict_classes(self, X, threshold=0.5):
        """Return class predictions using a threshold"""
        probabilities = self.predict(X)
        return (probabilities >= threshold).astype(int)

    def _sigmoid_func(self, x):
        return 1/(1 + np.exp(-x))

# Load the dataset
data = pd.read_csv("dataset_3classes_3_cat.csv")
# Separate features (X) and target (y)
X = data.iloc[:, :-1].values  # Features (all columns except the density)
y = data.iloc[:, -1].values  # Target ('density')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

# Measure time for training
start_training_time = time.time()

# Train separate logistic regression models for each class
# regressor0 = LogisticRegressionModel()
regressor0=SimpleLogisticRegressionModel()
regressor0.fit(X_train, [1 if val == 0 else 0 for val in y_train])

# regressor1 = LogisticRegressionModel()
regressor1=SimpleLogisticRegressionModel()
regressor1.fit(X_train, [1 if val == 1 else 0 for val in y_train])

# regressor2 = LogisticRegressionModel()
regressor2=SimpleLogisticRegressionModel()
regressor2.fit(X_train, [1 if val == 2 else 0 for val in y_train])

end_training_time = time.time()
training_duration = end_training_time - start_training_time

# Measure time for testing
start_testing_time = time.time()

# Make predictions
predictions0 = regressor0.predict(X_test)
predictions1 = regressor1.predict(X_test)
predictions2 = regressor2.predict(X_test)

# Determine final predicted class
result = [
    max(range(3), key=lambda i: [predictions0, predictions1, predictions2][i][index])
    for index in range(len(predictions0))
]

end_testing_time = time.time()
testing_duration = end_testing_time - start_testing_time

### Accuracy on training data
# Measure accuracy on training data
predictions0_train = regressor0.predict(X_train)
predictions1_train = regressor1.predict(X_train)
predictions2_train = regressor2.predict(X_train)

result_train = [
    max(range(3), key=lambda i: [predictions0_train, predictions1_train, predictions2_train][i][index])
    for index in range(len(predictions0_train))
]

# Calculate overall accuracy
train_accuracy = accuracy_score(y_train, result_train)
test_accuracy = accuracy_score(y_test, result)

# Calculate Precision
train_precision = precision_score(y_train, result_train, average='weighted')
test_precision = precision_score(y_test, result, average='weighted')

# Calculate Recall
train_recall = recall_score(y_train, result_train, average='weighted')
test_recall = recall_score(y_test, result, average='weighted')

# Calculate F1-score
train_f1 = f1_score(y_train, result_train, average='weighted')
test_f1 = f1_score(y_test, result, average='weighted')

# Print training results
print("Training Results:")
print(f"Accuracy: {train_accuracy:.4f}")
print(f"Precision: {train_precision:.4f}")
print(f"Recall: {train_recall:.4f}")
print(f"F1-score: {train_f1:.4f}")
print(f"Training Runtime: {training_duration:.4f} seconds\n")

# Print testing results
print("Testing Results:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1-score: {test_f1:.4f}")
print(f"Testing Runtime: {testing_duration:.4f} seconds\n")

# Print confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, result)
class_report = classification_report(y_test, result)

print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Print time measurements

print("\nTraining Time (seconds):", training_duration)
print("Testing Time (seconds):", testing_duration)
