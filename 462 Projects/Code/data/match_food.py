import pandas as pd
import json

# Load datasets
df_refined = pd.read_csv("dataset_full_refined.csv")
df_status = pd.read_csv("dataset_with_status.csv")
df_status= df_status.drop(columns=['main_dish'])
df_status= df_status.drop(columns=['vegetarian'])
# Load category JSON
with open("cat.json", "r", encoding="utf-8") as file:
    categories = json.load(file)

with open("vegan.json", "r", encoding="utf-8") as file:
    vcat = json.load(file)

# Create a reverse mapping from dish to category index
dish_to_category = {}
for index, (category, dishes) in enumerate(categories.items()):
    for dish in dishes:
        dish_to_category[dish] = index

# Create a reverse mapping from dish to category index
vdish_to_category = {}
for index, (category, dishes) in enumerate(vcat.items()):
    for dish in dishes:
        vdish_to_category[dish] = index

# Add new column `main_dish_cat`
def find_category(main_dish):
    # Ensure main_dish is a string and handle non-string values like NaN or floats
    if isinstance(main_dish, str):
        return dish_to_category.get(main_dish.lower(), 10)  # Default to -1 if not found
    else:
        # print(main_dish)
        return 10 # max cat number
    
# Add new column `main_dish_cat`
def find_vcategory(main_dish):
    # Ensure main_dish is a string and handle non-string values like NaN or floats
    if isinstance(main_dish, str):
        return vdish_to_category.get(main_dish.lower(), -1)  # Default to -1 if not found
    else:
        # print(main_dish)
        return -1 # max cat number

# Merge datasets on `date` and `count` columns
merged_df = pd.merge(df_status, df_refined[['date', 'count', 'main_dish','vegetarian']], on=['date', 'count'], how='left')
print(merged_df.head)
# Apply category lookup
merged_df['main_dish_cat'] = merged_df['main_dish'].apply(find_category)
merged_df['vegetarian_cat'] = merged_df['vegetarian'].apply(find_vcategory)
# Save the updated dataset
merged_df.to_csv("dataset_with_status_updated.csv", index=False)

print("New column 'main_dish_cat' added successfully. Updated dataset saved as 'dataset_with_status_updated.csv'.")

# # Specify the column name
# column_name = 'vegetarian'

# # Write the column contents to a text file
# with open("column_contents_vegetarian.txt", "w", encoding="utf-8") as file:
#     for value in df_refined[column_name].unique():
#         file.write(str(value) + "\n")

# print(f"Column '{column_name}' content written to 'column_contents.txt'.")
