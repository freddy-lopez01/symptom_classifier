import pandas as pd
import numpy as np
import csv




def main():
    severity_dic ={}
    with open('Symptom-severity.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            name, weight = row[0].split(",")
            name = name.strip().replace(" ", "")
            if weight == "weight":
                continue 
            severity_dic[name] = int(weight)

    #counter = 0
    column = []
    for key in severity_dic:
        #print(f"{key}: {severity_dic[key]}")
        column.append(key.strip().replace(" ", ""))
        #counter +=1
    #print(column)
    #print(counter)

    df = pd.DataFrame(columns=["Disease"] + column)
    original = pd.read_csv('dataset.csv')
    for index, row in original.iterrows():
        print("------------")
        print(f"index: {index}")
        print(row)
        print(f"row val: {row.values}")
        new_row = {symptom: 0 for symptom in column}
        print(f"first ----- {row["Disease"]}")
        new_row["Disease"] = row["Disease"]
        print(f"Disease {row["Disease"]}")
        print(f"index {index}")
        # Check symptoms in the old CSV row and set corresponding columns to 1
        df.loc[index, "Disease"] = row["Disease"]
        for symptom in row.values[1:]:
            if type(symptom) != float:
                print(symptom)
                symptom = symptom.strip().replace(" ", "")
                print(index)
                df.loc[index, symptom] = severity_dic[symptom]
        # Append the new row to the new DataFrame
        df._append(new_row, ignore_index=True)
    df.to_csv('reformated_dataset.csv', index=False)
    #with open('dataset.csv', newline='') as csvfile:
    #    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    #    for row in reader:
    #        print(row)
if __name__ == "__main__":
    main()
