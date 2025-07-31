import json
from lxml import html
import requests

def extract_data(url):
    response = requests.get(url, verify=False)
    # Parse the HTML
    tree = html.fromstring(response.text)
    menu_data = []
    for day in tree.xpath('//td[@class="single-day past"]'):
        # Extract the date and day of the week
        date = day.xpath('@data-date')[0]
        day_of_week = day.xpath('@headers')[0]
        
        # Extract meal information (lunch and dinner)
        meals = []
        for meal in day.xpath('.//div[@class="view-item view-item-aylik_menu"]'):
            meal_info = {}
            
            # Extract meal time (lunch or dinner)
            meal_time = meal.xpath('.//div[@class="views-field views-field-field-yemek-saati"]/div/text()')
            if meal_time:
                meal_info['meal_time'] = meal_time[0].strip()

            # Extract meal items (e.g., soup, main dish, side dishes, etc.)
            meal_info['soup'] = meal.xpath('.//div[@class="views-field views-field-field-ccorba"]/div/a/text()')
            meal_info['main_dish'] = meal.xpath('.//div[@class="views-field views-field-field-anaa-yemek"]/div/a/text()')
            meal_info['vegetarian'] = meal.xpath('.//div[@class="views-field views-field-field-vejetarien"]/div/a/text()')
            meal_info['side_dishes'] = meal.xpath('.//div[@class="views-field views-field-field-yardimciyemek"]/div/a/text()')
            meal_info['dessert'] = meal.xpath('.//div[@class="views-field views-field-field-aperatiff"]/div/a/text()')

            meals.append(meal_info)
        # Store the information for each day
        menu_data.append({
            'date': date,
            'day_of_week': day_of_week,
            'meals': meals
        })
    return menu_data

# Turkish to English character mapping
turkish_to_english = {
    'Ç': 'c', 'Ğ': 'g', 'I': 'i', 'İ': 'i', 'Ö': 'o', 'Ş': 's', 'Ü': 'u',
    'ç': 'c', 'ğ': 'g', 'ı': 'i', 'i': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u'
}

def convert_turkish_to_english(text):
    return ''.join(turkish_to_english.get(char, char) for char in text).lower()

def process_data(data):
    if isinstance(data, list):
        return [process_data(item) for item in data]
    elif isinstance(data, dict):
        return {convert_turkish_to_english(key): process_data(value) for key, value in data.items()}
    elif isinstance(data, str):
        return convert_turkish_to_english(data)
    return data  

if __name__ == '__main__':
    menu_data=[]
    # for i in range(1, 10):
    #     raw_data = extract_data(f"https://yemekhane.bogazici.edu.tr/aylik-menu/2024-0{i}")
    #     processed_data = process_data(raw_data)
    #     menu_data += processed_data

    raw_data = extract_data(f"https://yemekhane.bogazici.edu.tr/aylik-menu/2024-09")
    processed_data = process_data(raw_data)
    menu_data += processed_data
    raw_data = extract_data(f"https://yemekhane.bogazici.edu.tr/aylik-menu/2024-10")
    processed_data = process_data(raw_data)
    menu_data += processed_data
    with open('menu_data_910.json', 'w',  encoding='utf-8') as json_file:
        json.dump(menu_data, json_file, indent=4)

    # with open('menu_data.json', 'w',  encoding='utf-8') as json_file:
    #     json.dump(menu_data, json_file, ensure_ascii=False, indent=4)
