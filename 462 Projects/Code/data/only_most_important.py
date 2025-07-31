import pandas as pd

# Specify the input CSV file
input_file = "dataset_dorms.csv"

# Specify the output CSV file
output_file = "important_data.csv"

# List of columns to keep
columns_to_keep = ['campus','weather','status','density']  # Replace with your columns

# Read the CSV file
df = pd.read_csv(input_file)

# Keep only the specified columns
filtered_df = df[columns_to_keep]

# Save the filtered data to a new CSV file
filtered_df.to_csv(output_file, index=False)

print(f"Filtered data saved to {output_file}")
