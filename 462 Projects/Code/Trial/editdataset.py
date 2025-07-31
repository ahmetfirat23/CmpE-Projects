import pandas as pd

def process_dataset(file_path, output_path):
    # Read the dataset
    df = pd.read_csv(file_path)

    # Extract the month from the date column
    df['month'] = pd.to_datetime(df['date']).dt.month

    # Create indexes for unique values in categorical columns
    day_index = {day: i for i, day in enumerate(df['day'].unique())}
    meal_index = {meal: i for i, meal in enumerate(df['meal'].unique())}
    soup_index = {soup: i for i, soup in enumerate(df['soup'].unique())}
    main_dish_index = {main_dish: i for i, main_dish in enumerate(df['main_dish'].unique())}
    vegetarian_index = {veg: i for i, veg in enumerate(df['vegetarian'].unique())}
    month_index = {month: i for i, month in enumerate(df['month'].unique())}

    # Map the indexes back to the dataset
    df['day_index'] = df['day'].map(day_index)
    df['meal_index'] = df['meal'].map(meal_index)
    df['soup_index'] = df['soup'].map(soup_index)
    df['main_dish_index'] = df['main_dish'].map(main_dish_index)
    df['vegetarian_index'] = df['vegetarian'].map(vegetarian_index)
    df['month_index'] = df['month'].map(month_index)

    # Create a new dataset with only indexes and count
    new_df = df[['month_index', 'day_index', 'meal_index', 'soup_index', 'main_dish_index', 'vegetarian_index', 'count']]

    # Save the new dataset to a CSV file
    new_df.to_csv(output_path, index=False)

    # Return index mappings for reference
    return {
        'day_index': day_index,
        'meal_index': meal_index,
        'soup_index': soup_index,
        'main_dish_index': main_dish_index,
        'vegetarian_index': vegetarian_index,
        'month_index': month_index,
    }


# Example usage
file_path = 'dataset_full_refined.csv'  # Replace with the path to your dataset
output_path = 'new_dataset.csv'  # Replace with the desired output path

indexes = process_dataset(file_path, output_path)
print(indexes)
