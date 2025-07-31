import os
import sys
import json
import time

PAGE_SIZE = 10
FILE_SIZE = 100
MAXIMUM_NUM_FIELDS = 15
MAXIMUM_LEN_TYPE_NAME = 30
MAXIMUM_LEN_FIELD_NAME = 30


# Creates a new type with the given name and fields
def create_type(catalogue, type_name, args):
    number_of_fields = args[0]
    try:
        primary_key_order = args[1]
    except:
        raise Exception
    # If there is not enough fields
    if len(args) != 2 + int(number_of_fields) * 2:
        raise Exception
    # Type name can be at most MAXIMUM_LEN_TYPE_NAME characters
    if len(type_name) > MAXIMUM_LEN_TYPE_NAME:
        raise Exception
    # If type already exists
    if type_name in catalogue:
        raise Exception
    # Create a list of tuples for each field (field_name, field_type)
    pairs = [(args[i], args[i+1]) for i in range(2, len(args), 2)]
    # There can be at most MAXIMUM_NUM_FIELDS fields
    if len(pairs) > MAXIMUM_NUM_FIELDS:
        raise Exception

    for pair in pairs:
        # Field name can be at most MAXIMUM_LEN_FIELD_NAME characters
        if len(pair[0]) > MAXIMUM_LEN_FIELD_NAME:
            raise Exception
        # If field type doesn't exist
        if pair[1] not in ['int', 'str']:
            raise Exception
    # Add the type to the catalogue
    catalogue[type_name] = {
        'primary_key': pairs[int(primary_key_order) - 1][0],
        'fields': pairs
    }
    # Create a file for the type
    # Directories are regarded as files!!!
    os.makedirs(type_name)
    # Initialize the pages with empty dictionaries
    for i in range(0, FILE_SIZE):
        with open(f"{type_name}/page_{i}.json", 'w') as f:
            json.dump({}, f)
    # Create the lookup table
    with open(f"{type_name}/lookup.json", 'w') as f:
        json.dump({}, f)
    # Create the metadata file
    record_counts = {}
    for i in range(0, FILE_SIZE):
        record_counts[i] = 0

    with open(f"{type_name}/metadata.json", 'w') as f:
        json.dump({
            'record_counts': record_counts,
        }, f)
    # Dump the catalogue
    save_catalogue(catalogue)
    return


# Creates a new record with the given type and fields
def create_record(catalogue, type_name, args):
    # If type doesn't exist
    if type_name not in catalogue:
        raise Exception
    # If there are not enough fields
    if len(args) != len(catalogue[type_name]['fields']):
        raise Exception
    # Create the record
    record = {}
    for i in range(len(args)):
        # Checks if the integer input is valid
        if catalogue[type_name]['fields'][i][1] == 'int':
            try:
                record[catalogue[type_name]['fields'][i][0]] = int(args[i])
            except:
                raise Exception
        # Save the value to the field in the record
        record[catalogue[type_name]['fields'][i][0]] = args[i]
    
    # Check if the record already exists by checking the lookup table with primary key

    primary_key = record[catalogue[type_name]['primary_key']]
    with open(f"{type_name}/lookup.json", 'r') as f:
        lookup = json.load(f)
    if primary_key in lookup:
        raise Exception
    
    # Find the page to insert the record
    with open(f"{type_name}/metadata.json", 'r') as f:
        metadata = json.load(f)
    # loop through the pages to find the first page with less than PAGE_SIZE records
    record_counts = metadata['record_counts']
    for i in range(0, FILE_SIZE):
        if record_counts[str(i)] < PAGE_SIZE:
            # Open the found page and insert the record
            with open(f"{type_name}/page_{i}.json", 'r') as f:
                page = json.load(f)
            page[primary_key] = record
            # Rewrite the page with the new record
            with open(f"{type_name}/page_{i}.json", 'w') as f:
                json.dump(page, f)
            record_counts[str(i)] += 1
            break
    # Update the metadata and lookup table
    metadata['record_counts'] = record_counts
    with open(f"{type_name}/metadata.json", 'w') as f:
        json.dump(metadata, f)
    lookup[primary_key] = i
    with open(f"{type_name}/lookup.json", 'w') as f:
        json.dump(lookup, f)
    return


# Deletes a record with the given primary key
def delete_record(catalogue, type_name, args):
    # If type doesn't exist
    if type_name not in catalogue:
        raise Exception
    primary_key = args[0]
    with open(f"{type_name}/lookup.json", 'r') as f:
        lookup = json.load(f)
    # If record doesn't exist
    if primary_key not in lookup:
        raise Exception
    # Find which page the record is in
    page_number = lookup[primary_key]
    # Open the page and delete the record
    with open(f"{type_name}/page_{page_number}.json", 'r') as f:
        page = json.load(f)
    del page[primary_key]
    # Rewrite the page without the deleted record
    with open(f"{type_name}/page_{page_number}.json", 'w') as f:
        json.dump(page, f)
    # Update the metadata and lookup table
    with open(f"{type_name}/metadata.json", 'r') as f:
        metadata = json.load(f)
    record_counts = metadata['record_counts']
    record_counts[str(page_number)] -= 1
    metadata['record_counts'] = record_counts
    with open(f"{type_name}/metadata.json", 'w') as f:
        json.dump(metadata, f)
    del lookup[primary_key]
    with open(f"{type_name}/lookup.json", 'w') as f:
        json.dump(lookup, f)
    return


# Searches for a record with the given primary key
def search_record(catalogue, type_name, args):
    # If type doesn't exist
    if type_name not in catalogue:
        raise Exception
    
    primary_key = args[0]
    with open(f"{type_name}/lookup.json", 'r') as f:
        lookup = json.load(f)
    # If record doesn't exist
    if primary_key not in lookup:
        raise Exception
    # Find which page the record is in
    page_number = lookup[primary_key]
    # Open the page and print the record
    with open(f"{type_name}/page_{page_number}.json", 'r') as f:
        page = json.load(f)
    fields_arr = []
    for field in catalogue[type_name]['fields']:
        fields_arr.append(page[primary_key][field[0]])
    # Log the output to output.txt
    with open("output.txt", 'a') as f:
        f.write(' '.join(fields_arr) + "\n")
    return


# Processes a command and logs the result
def process_command(catalogue, command):
    # Get unix timestamp
    command_time = int(time.time())
    try:
        args = command.split()
        # No command has less than 4 arguments
        if len(args) < 4:
            raise Exception
        
        # Find the command and call the appropriate function
        type_name = args[2]
        if args[0] == 'create':
            if args[1] == 'type':
                create_type(catalogue, type_name, args[3:])
            elif args[1] == 'record':
                create_record(catalogue, type_name, args[3:])
        elif args[0] == 'delete':
            if args[1] == 'record':
                delete_record(catalogue, type_name, args[3:])
        elif args[0] == 'search':
            if args[1] == 'record':
                search_record(catalogue, type_name, args[3:])
        else:
            raise Exception
        # Log the command and result
        with open("log.txt", 'a') as f:
            f.write(f"{command_time}, {command.strip()}, success\n")
    except:
        # Log the command and failure
        with open("log.txt", 'a') as f:
            f.write(f"{command_time}, {command.strip()}, failure\n")


# Dumps the catalogue to the file
def save_catalogue(catalogue):
    json.dump(catalogue, open('catalogue.json', 'w'))


# Loads the catalogue from the file or returns an empty dictionary
def load_catalogue():
    try:
        return json.load(open('catalogue.json'))
    except:
        return {}


def main():
    try:
        # Read commands from file
        with open(sys.argv[1], 'r') as f:
            commands = f.readlines()
    except:
        # If input file not found, print error and stop execution
        return
    
    # Load catalogue that holds information about types (primary key and fields)
    catalogue = load_catalogue()
    # Process each command
    for command in commands:
        process_command(catalogue, command)


if __name__ == '__main__':
    main()