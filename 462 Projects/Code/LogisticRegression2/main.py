from LogisticRegression import LogisticRegressionModel
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import pandas as pd
import time

# Load the dataset
data = pd.read_csv("dataset_3classes_3_cat.csv")

# Balance the dataset
# Separate majority and minority classes
class_counts = data['density'].value_counts()
print("Class distribution before balancing:")
print(class_counts)
# Over sample
# max_samples = class_counts.max()
# balanced_data = pd.DataFrame()

# for class_label in class_counts.index:
#     class_data = data[data['density'] == class_label]
#     class_data_balanced = resample(
#         class_data, 
#         replace=True,  # Oversample with replacement
#         n_samples=max_samples,  # Match the maximum class count
#         random_state=42
#     )
#     balanced_data = pd.concat([balanced_data, class_data_balanced])
# under sample
# min_samples = class_counts.min()
# balanced_data = pd.DataFrame()

# for class_label in class_counts.index:
#     class_data = data[data['density'] == class_label]
#     class_data_balanced = resample(
#         class_data, 
#         replace=False,  # Undersample without replacement
#         n_samples=min_samples,  # Match the minimum class count
#         random_state=1234
#     )
#     balanced_data = pd.concat([balanced_data, class_data_balanced])
# # Shuffle the balanced dataset
# balanced_data = balanced_data.sample(frac=1, random_state=1234).reset_index(drop=True)
# print("Class distribution after balancing:")
# print(balanced_data['density'].value_counts())

# Drop specific columns from X before using it
# columns_to_drop = ['day_of_week', 'week_of_month','month', 'soup', 'side_dishes']  # Specify columns to drop
# X = balanced_data.drop(columns=columns_to_drop).iloc[:, :-1].values  # Features (excluding dropped columns and target)

balanced_data=data
# Separate features (X) and target (y)
X = balanced_data.iloc[:, :-1].values  # Features (all columns except the last one)
y = balanced_data.iloc[:, -1].values  # Target ('density')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

# Measure time for training
start_training_time = time.time()

# Train separate logistic regression models for each class
regressor0 = LogisticRegressionModel()
regressor0.fit(X_train, [1 if val == 0 else 0 for val in y_train])

regressor1 = LogisticRegressionModel()
regressor1.fit(X_train, [1 if val == 1 else 0 for val in y_train])

regressor2 = LogisticRegressionModel()
regressor2.fit(X_train, [1 if val == 2 else 0 for val in y_train])

end_training_time = time.time()
training_duration = end_training_time - start_training_time

# Measure time for testing
start_testing_time = time.time()

# Make predictions
predictions0 = regressor0.predict(X_test)
predictions1 = regressor1.predict(X_test)
predictions2 = regressor2.predict(X_test)

# Aggregate predictions and determine final predicted class
result = [
    max(range(3), key=lambda i: [predictions0, predictions1, predictions2][i][index])
    for index in range(len(predictions0))
]


end_testing_time = time.time()
testing_duration = end_testing_time - start_testing_time
### accuracy on training data
# Measure accuracy on training data
predictions0_train = regressor0.predict(X_train)
predictions1_train = regressor1.predict(X_train)
predictions2_train = regressor2.predict(X_train)

result_train = [
    max(range(3), key=lambda i: [predictions0_train, predictions1_train, predictions2_train][i][index])
    for index in range(len(predictions0_train))
]
# Calculate overall accuracy
training_accuracy = accuracy(y_train, result_train)
overall_accuracy = accuracy(y_test, result)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, result)

# Generate classification report
class_report = classification_report(y_test, result, target_names=["Class 0 : Low", "Class 1: Medium", "Class 2: High"])

# Print the detailed report
print("Training Accuracy:", training_accuracy)
print("Testing Accuracy:", overall_accuracy)
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Print time measurements
print("\nTraining Time (seconds):", training_duration)
print("Testing Time (seconds):", testing_duration)
