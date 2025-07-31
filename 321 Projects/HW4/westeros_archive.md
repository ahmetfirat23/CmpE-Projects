















# Westeros Archive
### CMPE321 - Project 4

Ahmet Fırat Gamsız – 2020400180
Yunus Emre Özdemir - 2020400153
















1. General Design Choices
2. Execution
3. General Flow
4. Design & Implementation of Operations
	1. Creating a Type
	2. Creating a Record
	3. Searching a Record
	4. Deleting a Record
5. Assumptions and Limitations
6. Further Improvements and Setbacks


## 1 General Design Choices
#### Definitions
* Page Size: 10 records
* File Size: 100 pages

#### Implement a Catalogue
We have designed a Catalogue where we keep primary_key and field information about types.
Example:
```json
{
  "human": {
    "primary_key": "name",
    "fields": [
      ["name", "str"],
      ["origin", "str"],
      ["title", "str"],
      ["age", "int"],
      ["weapon", "str"],
      ["skill", "str"]
    ]
  }
}
```
This is the only file we always keep in the memory. We load this at the `main()` function from `catalogue.json` at the start and write to it as catalogue changes in the `create_type() `function. We wrote utility functions `load_catalogue()` and `save_catalogue()` for this purposes.
With this catalogue, we can easily check if a type exists and find out its primary key and fields. We can also preserve the state of our database and load it at the start of execution.

#### Implement Types as Folders, Pages as JSON files
In our implementation, only a single page is loaded to the memory regardless of the type of the command. Details of this can be found in 4 Design & Implementation of Operations part. Therefore, by keeping all pages in separate files we can achieve better abstraction where we don’t need to be concerned about file handler location calculations. By keeping pages as JSON files and having primary keys of records as keys we increase the access time to records since there is no need to traverse all records in a page.
Example Type: (In the real implementation we have pages up to 100 - File Size)
```
human (folder)
- lookup.json
- metadata.json
- page_0.json
- page_1.json
```

Example Page:
```json
{
  "RamsayBolton": {
    "name": "RamsayBolton",
    "origin": "Dreadfort",
    "title": "Lord",
    "age": "21",
    "weapon": "Dagger",
    "skill": "Strategy"
  }
}
```

#### Keep Metadata for Every Type
Within every type folder, we keep a `metadata.json` file to keep track of the amount of records for each page.
Example: (In the real implementation we have keys up to 100 - File Size)
```json
{
  "record_counts": {
    "0": 2,
    "1": 0,
    "2": 0,
	}
}
```
Using this metadata, we don’t need to load and check every page to see if there is room for a new record. Therefore, our insertions become faster.

#### Keep a Lookup Table for Every Type
Within every type folder, we keep a `lookup.json` file which is basically a lookup table to find in which page the record resides by their primary key.
Example:
```json
{
	"RamsayBolton": 0,
	"Bronn": 0
}
```
This makes searching and deleting a record constant time.

## 2 Execution
``` 
python3 2020400180_2020400153/archive.py <fullinputfilepath>
```
* The program will generate all files related to execution along with the `output.txt` and `log.txt` at the folder in which you are running it from.

## 3 General Flow
1. Program starts with the `main()` function. Path of the input file is taken from the command line arguments, commands are read from the input file, catalogue data is loaded with `load_catalogue()` function and finally `process_command()` is called for every command.
2. `process_command()` function calls the respective function for each command. Additionally, it catches errors and writes the status of the operations to log.txt.
3. `create_type()`, `create_record()`, `search_record()` and `delete_record()` functions execute the given command and write to `output.txt` if necessary. These functions are explained in detail at 4 Design & Implementation of Operations part.

## 4 Design & Implementation of Operations
### 4.1 Creating a Type
Creating a Type is implemented in `create_type()` function.
**Complexity:** Constant time
1. Arguments are parsed and information about the new type is inserted into the in-memory catalogue.
2. If a type already exists in the catalogue an exception is thrown.
3. Folder for the type, `metadata.json`, `lookup.json` and pages are initialized.
4. Catalogue is also saved to `catalogue.json` at every creation.

### 4.2 Creating a Record
Creating a Record is implemented in `create_record() `function.
**Complexity:** Linear time (array traversal, only single page load happens)
1. Type name is checked in the catalogue, if it doesn’t exist and exception is thrown.
2. Primary key of the new record is looked up in `lookup.json` of the type to check if it already exists. 
3. `record_counts` in `metadata.json` is traversed to find an available page.
4. Record is inserted in the page, `lookup.json` and `metadata.json` are updated.

### 4.3 Searching a Record
Searching a Record is implemented in `search_record()` function.
**Complexity:** Constant time
1. Type name is checked in the catalogue, if it doesn’t exist and exception is thrown.
2. `lookup.json` of the type is loaded and the page index of the record is found.
3. Page is loaded, record is found in the page using its primary key and returned.

### 4.4 Deleting a Record
Deleting a Record is implemented in `delete_record()` function.
**Complexity:** Constant time
1. Type name is checked in the catalogue, if it doesn’t exist and exception is thrown.
2. Page of the record is found in the same way as searching a record.
3. Record is deleted from the page, `metadata.json` and `lookup.json` are updated.

## 5 Assumptions and Limitations
* We assumed maximum 1000 entries can be stored for a type. A new file won’t be created when 1001th record inserted. We didn’t include this because it has been said that it won’t be tested. A solution for this can be doubling the amount of page files after a certain percentage of slots are full.
* We made extra checks for given assumptions in the description. They were used for our testing purposes. (Such as checking if integer fields can actually be casted to integers.)

## 6 Further Improvements and Setbacks
* Hashing primary keys to page indices to insert and search for records using quadratic probing and valid/invalid flags can make sense. We preferred using lookup tables since it is a simple and adequate implementation.
