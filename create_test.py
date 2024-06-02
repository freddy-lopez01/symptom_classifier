import random
import csv

list_sym = []
with open('dataset.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        name = row[0]
        if "Disease" in name:
            continue 
        for sym in row[1::]:
            if sym not in list_sym:
                list_sym.append(sym)
            
# Step 1: Define the list of 134 unique strings
unique_strings = list_sym

# Step 2: Generate the data with 100 rows and 8 columns
num_rows = 100
num_columns = 8

data = [
    [random.choice(unique_strings) for _ in range(num_columns)]
    for _ in range(num_rows)
]

# Step 3: Write the data to a CSV file
filename = "random_strings.csv"

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

print(f"CSV file '{filename}' created successfully.")

