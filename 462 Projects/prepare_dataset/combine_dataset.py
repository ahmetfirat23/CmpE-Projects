import csv

days_of_week={
"sali":"tuesday",
"carsamba":"wednesday",
"persembe":"thursday",
"cuma":"friday",
"cumartesi":"saturday",
"pazar":"sunday",
"pazartesi":"monday"
}

meals={
    "kahvalti yemegi":"breakfast",
    "ogle yemegi":"lunch",
    "aksam yemegi":"dinner"
}

def combine_datasets():
	final_dataset = [
		["date", "day", "campus", "weather", "count", "meal" , "soup","main_dish", "vegetarian","side_dishes","dessert"]
	]
	data=[]
	menu=[]
	menu_breakfast=[]
	with open('./data/meal_data_910.csv') as file_obj: 
		reader_data = csv.reader(file_obj) 
		for item in reader_data:
			data.append(item)
	with open('./data/menu_data_910.csv') as file_obj: 
		reader_menu = csv.reader(file_obj) 
		for item in reader_menu:
			menu.append(item)
	with open('./data/menu_data_breakfast_910.csv') as file_obj: 
		reader_menu_breakfast = csv.reader(file_obj) 
		for item in reader_menu_breakfast:
			menu_breakfast.append(item)
	"""
	data.csv: date, campus(tr), meal(en), weather, count
	menu_data.csv: date,day_of_week (tr),meal_time (tr),soup,main_dish,vegetarian,side_dishes,dessert > dishes in tr
	menu_data_breakfast.csv: date,day_of_week (tr),meal_time(tr),soup,main_dish,vegetarian,side_dishes,dessert > dishes in tr
	dataset: date, day, campus, weather, count,meal ,soup,main_dish,vegetarian,side_dishes,dessert
	"""
	# lunch and dinner
	for item in data[1:]:
		for row in menu[1:]:
			if row[0]==item[0] and meals[row[2]]==item[2]: #same day and same meal
				final_dataset.append([item[0], days_of_week[row[1]], item[1],item[3], item[4], item[2]]+row[3:])
	# breakfast			
	for item in data[1:]:
		for row in menu_breakfast[1:]:
			if row[0]==item[0] and meals[row[2]]==item[2]: #same day and same meal
				final_dataset.append([item[0], days_of_week[row[1]], item[1],item[3], item[4], item[2]]+row[3:])
	return final_dataset
	


def combine_only_breakfast():
	final_dataset = [
		["date", "day", "campus", "weather", "count", "meal"]
	]
	data=[]
	menu=[]
	menu_breakfast=[]
	with open('./data/meal_data_910.csv') as file_obj: 
		reader_data = csv.reader(file_obj) 
		for item in reader_data:
			data.append(item)
	with open('./data/menu_data_910.csv') as file_obj: 
		reader_menu = csv.reader(file_obj) 
		for item in reader_menu:
			menu.append(item)
	with open('./data/menu_data_breakfast_910.csv') as file_obj: 
		reader_menu_breakfast = csv.reader(file_obj) 
		for item in reader_menu_breakfast:
			menu_breakfast.append(item)
	"""
	data.csv: date, campus(tr), meal(en), weather, count
	menu_data.csv: date,day_of_week (tr),meal_time (tr),soup,main_dish,vegetarian,side_dishes,dessert > dishes in tr
	menu_data_breakfast.csv: date,day_of_week (tr),meal_time(tr),soup,main_dish,vegetarian,side_dishes,dessert > dishes in tr
	dataset: date, day, campus, weather, count,meal
	"""
	for item in data[1:]:
		day=""
		for row in menu[1:]:
			if row[0]==item[0]:
				day=days_of_week[row[1]]
				break
		final_dataset.append([item[0], day, item[1],item[3], item[4], item[2]])

	return final_dataset



def write_to_csv(csv_file_path,data):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

if __name__ == '__main__':
	final_dataset=combine_datasets()
	csv_file_path = 'dataset_910.csv'  
	write_to_csv(csv_file_path,final_dataset) 
	#####
	breakfast_without_menu=combine_only_breakfast()
	csv_file_path = 'dataset_breakfast_910.csv'  
	write_to_csv(csv_file_path,breakfast_without_menu) 
