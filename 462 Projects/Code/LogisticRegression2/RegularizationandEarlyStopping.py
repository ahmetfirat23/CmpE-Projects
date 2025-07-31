import numpy as np

class LogisticRegressionModel:
    def __init__(self, lr=0.0001, num_iter=1000000, reg_strength=0, early_stopping_rounds=5, tol=0.001e-4):
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
        # Initialize weights as a column vector
        self.weight = np.zeros((num_features, 1))
        self.bias = 0
        
        # Ensure Y is a column vector
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
            # Ensure shapes are correct for matrix operations
            error = y_predicted - Y  # Shape: (N, 1)
            dw = (1/N) * np.dot(X.T, error) + (self.reg_strength / N) * self.weight  # Shape: (features, 1)
            db = (1/N) * np.sum(error)
            
            # Update parameters
            self.weight -= self.lr * dw
            self.bias -= self.lr * db
           
    def predict(self, X):
        linear_model = np.dot(X, self.weight) + self.bias
        y_predicted = self._sigmoid_func(linear_model)
        return y_predicted.reshape(-1)  # Flatten predictions for output

    def predict_classes(self, X, threshold=0.5):
        """Return class predictions using a threshold"""
        probabilities = self.predict(X)
        return (probabilities >= threshold).astype(int)

    def _sigmoid_func(self, x):
        return 1/(1 + np.exp(-x))