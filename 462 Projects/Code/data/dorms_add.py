import pandas as pd

# Load the CSV file
df = pd.read_csv('dataset_new_cat.csv')

# Define a mapping for the dorms based on campus values
campus_to_dorms = {
    0: 1,
    1: 2,
    2: 0,
    3: 1,
    4: 1,
    5: 2
}

# Create a new column 'dorms' based on the 'campus' column
df['dorms'] = df['campus'].map(campus_to_dorms)

# Switch the last two columns (status and density)
cols = df.columns.tolist()
cols[-2], cols[-1] = cols[-1], cols[-2]
df = df[cols]

# Save the modified dataframe to a new CSV file
df.to_csv('dataset_dorms.csv', index=False)

# Show the updated dataframe
print(df)
