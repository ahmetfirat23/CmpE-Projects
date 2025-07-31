import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset
df = pd.read_csv('dataset_dorms.csv')

# Add 2 random columns
np.random.seed(42)  # for reproducibility
df['random_col1'] = np.random.randint(0, 100, size=len(df))
df['random_col2'] = np.random.randint(0, 100, size=len(df))

# # Create categorical target variable based on count ranges
# def categorize_count(count):
#     if count <= 46.97:
#         return 'y0'
#     elif count <=  371.00:
#         return 'y1'
#     elif count <= 1517.27:
#         return 'y2'
#     else:
#         return 'y3'

# df['count_category'] = df['count'].apply(categorize_count)
# campus,weather,meal,soup,side_dishes,dessert,day_of_week,week_of_month,month,vegetarian_cat,main_dish_cat,status,density
# Prepare features and target
X = df[['campus', 'weather', 'meal', 'soup', 
        'side_dishes', 'dessert','day_of_week','week_of_month', 'month','vegetarian_cat','main_dish_cat','status', 'dorms', 'random_col1', 'random_col2']]
y = df['density']

# Encode target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y_encoded)

# Get feature importance scores
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Create visualization
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Feature Importance Analysis')
plt.xlabel('Importance Score')
plt.ylabel('Features')

# Print feature importance scores
print("\nFeature Importance Scores:")
print(feature_importance.to_string(index=False))

# Print summary statistics
print("\nCount Distribution:")
print(df['density'].value_counts())