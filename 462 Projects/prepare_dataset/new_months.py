import pandas as pd
import re

def normalize_turkish_chars(text):
    """
    Normalize Turkish characters to their ASCII equivalents
    """
    char_map = {
        'ç': 'c', 
        'ğ': 'g', 
        'ı': 'i', 
        'ö': 'o', 
        'ş': 's', 
        'ü': 'u'
    }
    return ''.join(char_map.get(char.lower(), char) for char in text)

def convert_excel_to_csv(input_file, output_file):
    """
    Convert Excel file to CSV with specified requirements
    """
    # Read the Excel file
    df = pd.read_excel(input_file)
    
    # Create a mapping for Turkish meal names to English
    meal_map = {
        'KAHVALTI': 'breakfast',
        'ÖĞLE YEMEĞİ': 'lunch', 
        'AKŞAM YEMEĞİ': 'dinner'
    }
    
    # Prepare the final dataframe
    results = []
    
    for _, row in df.iterrows():
        # Normalize campus name
        campus = normalize_turkish_chars(row['Yer'])
        
        # Determine meal type and count
        meal = None
        count = None
        
        for meal_turkish, meal_english in meal_map.items():
            if not pd.isna(row.get(meal_turkish)):
                meal = meal_english
                count = row[meal_turkish]
                break
        
        # Skip if no meal or count found
        if not meal or pd.isna(count):
            continue
        
        results.append({
            'Date': row['Tarih'],
            'campus': campus,
            'meal': meal,
            'count': int(count)
        })
    
    # Create new dataframe and save to CSV
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_file, index=False)
    
    print(f"Converted data saved to {output_file}")

# Example usage
convert_excel_to_csv('newmonths.xlsx', 'meal_data.csv')