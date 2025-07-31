import json
import csv

# Specify the input JSON file and output CSV file
input_json_file = 'menu_data_910.json'
output_csv_file = 'menu_data_910.csv'

# Open the JSON file and load the data
with open(input_json_file, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# Open the CSV file for writing
with open(output_csv_file, 'w', newline='', encoding='utf-8') as csv_file:
    # Define the CSV headers
    headers = [
        "date", "day_of_week", "meal_time", 
        "soup", "main_dish", "vegetarian", 
        "side_dishes", "dessert"
    ]
    writer = csv.DictWriter(csv_file, fieldnames=headers)
    
    # Write the header row
    writer.writeheader()
    
    # Loop through the JSON data and write rows for each meal
    for day in data:
        date = day["date"]
        day_of_week = day["day_of_week"]
        for meal in day["meals"]:
            writer.writerow({
                "date": date,
                "day_of_week": day_of_week,
                "meal_time": meal["meal_time"],
                "soup": ", ".join(meal["soup"]),
                "main_dish": ", ".join(meal["main_dish"]),
                "vegetarian": ", ".join(meal["vegetarian"]),
                "side_dishes": ", ".join(meal["side_dishes"]),
                "dessert": ", ".join(meal["dessert"]),
            })

print(f"JSON data has been successfully converted to {output_csv_file}.")
