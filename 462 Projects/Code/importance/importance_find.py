import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Load the data from CSV
data = pd.read_csv('dataset_new.csv')

# Define the target variable and features
X = data.drop('density', axis=1)  # Features
y = data['density']  # Target variable

# Add two random features to the dataset
np.random.seed(1234)
X['random_feature_1'] = np.random.rand(len(X))  # Random feature 1
X['random_feature_2'] = np.random.rand(len(X))  # Random feature 2

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Train an XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=1234)
model.fit(X_train, y_train)

# Plot feature importance
xgb.plot_importance(model, importance_type='weight', max_num_features=15)
plt.show()
