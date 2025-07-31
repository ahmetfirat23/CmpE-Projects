import numpy as np

class LogisticRegressionModel:
    def __init__(self, lr=0.0001, num_iter=1000, reg_strength=0.1):
        self.lr=lr
        self.num_iter=num_iter
        self.weight=None
        self.bias=None
        self.reg_strength = reg_strength

    def fit(self, X, Y):
        N , num_features= X.shape
        self.weight=np.zeros(num_features)
        self.bias=0
        for i in range(self.num_iter):
            linear_model= np.dot(X,self.weight)+self.bias # wX+b
            y_predicted=self._sigmoid_func(linear_model) # Sigmoid of (wX+b)
            # Gradient Descent
            # # dw, db comes from min of cross entropy loss function
            # dw= (1/N) * np.dot(X.T, (y_predicted-Y)) 
            # db=(1/N)* sum(y_predicted-Y)
             # Gradient Descent with L2 Regularization
            dw = (1 / N) * np.dot(X.T, (y_predicted - Y)) + (self.reg_strength / N) * self.weight
            db = (1 / N) * np.sum(y_predicted - Y)


            self.weight -= self.lr*dw # update weight 
            self.bias -= self.lr*db # updated bias
           
    def predict(self,X):
        linear_model= np.dot(X,self.weight)+self.bias # wX+b
        y_predicted=self._sigmoid_func(linear_model) # Sigmoid of (wX+b)
        # Choose Threshold as 0.5 (prob>0.5 mean y=1; otherwise y=0)
        # y_class=[1 if i>0.5 else 0 for i in y_predicted]
        return y_predicted

    def _sigmoid_func(self,x):
        return 1/(1+np.exp(-1*x))



LogisticRegressionModel()


