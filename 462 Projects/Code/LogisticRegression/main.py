from LogisticRegression import LogisticRegressionModel
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import pandas as pd

# Load the dataset
data = pd.read_csv("dataset_new_cat.csv")

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

# Drop specific columns from X before using it
columns_to_drop = ['day_of_week', 'week_of_month','month', 'soup', 'side_dishes']  # Specify columns to drop
X = balanced_data.drop(columns=columns_to_drop).iloc[:, :-1].values  # Features (excluding dropped columns and target)

# Separate features (X) and target (y)
# X = balanced_data.iloc[:, :-1].values  # Features (all columns except the last one)
y = balanced_data.iloc[:, -1].values  # Target ('density')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


#  lr=0.0001, num_iter=10000, reg_strength=0.1
# Train separate logistic regression models for each class
regressor0 = LogisticRegressionModel(lr=0.0001, num_iter=5000)
regressor0.fit(X_train, [1 if val == 0 else 0 for val in y_train])
predictions0 = regressor0.predict(X_test)

regressor1 = LogisticRegressionModel(lr=0.0001, num_iter=5000)
regressor1.fit(X_train, [1 if val == 1 else 0 for val in y_train])
predictions1 = regressor1.predict(X_test)

regressor2 = LogisticRegressionModel(lr=0.0001, num_iter=5000)
regressor2.fit(X_train, [1 if val == 2 else 0 for val in y_train])
predictions2 = regressor2.predict(X_test)

# Aggregate predictions and determine final predicted class
result = [
    max(range(3), key=lambda i: [predictions0, predictions1, predictions2][i][index])
    for index in range(len(predictions0))
]

# Calculate overall accuracy
overall_accuracy = accuracy(y_test, result)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, result)

# Generate classification report
class_report = classification_report(y_test, result, target_names=["Class 0", "Class 1", "Class 2"])

# Print the detailed report
print("Overall Accuracy:", overall_accuracy)
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)



