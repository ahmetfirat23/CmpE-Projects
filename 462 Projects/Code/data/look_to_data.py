import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('dataset_dorms.csv')

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# Descriptive statistics
print("\nDescriptive Statistics:")
print(data.describe())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Frequency of 'density'
print("\nDensity Value Counts:")
print(data['density'].value_counts())

# Correlation matrix
correlation_matrix = data.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Visualization 1: Density distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='density', palette='viridis')
plt.title('Density Distribution')
plt.xlabel('Density')
plt.ylabel('Frequency')
plt.show()

# Visualization 2: Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Visualization 3: Box plot of 'weather' vs 'density'
plt.figure(figsize=(8, 6))
sns.boxplot(data=data, x='density', y='weather', palette='muted')
plt.title('Weather Distribution by Density')
plt.xlabel('Density')
plt.ylabel('Weather')
plt.show()

# Visualization 4: Side dishes vs dessert
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='side_dishes', y='dessert', hue='density', palette='deep')
plt.title('Side Dishes vs Dessert Colored by Density')
plt.xlabel('Side Dishes')
plt.ylabel('Dessert')
plt.show()
