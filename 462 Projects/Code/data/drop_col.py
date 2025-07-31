import pandas as pd

# Load the dataset
file_path = 'dataset_with_status_updated.csv'  # Replace with the correct path to your data file
data = pd.read_csv(file_path)




# Drop specified columns
columns_to_drop = ['date', 'day', 'count', 'normalized_count', 'main_dish','vegetarian']
new_data = data.drop(columns=columns_to_drop)
columns = list(new_data.columns)
columns[-1], columns[-2], columns[-3], columns[-4] = columns[-4], columns[-3], columns[-2], columns[-1]
new_data = new_data[columns]
density_mapping = {
    'Very Low': 0,
    'Low': 0,
    'Medium': 1,
    'High': 2,
    'Very High': 2
}
# Replace the density column values
new_data['density'] = data['density'].map(density_mapping)
# Save the new dataset
new_data.to_csv('dataset_new_cat.csv', index=False)

print("The new dataset has been saved as 'new_dataset.csv'.")
