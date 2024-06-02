import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import csv 

def train_model():
    df = pd.read_csv('dataset.csv')
    #df.describe()
    df1 = pd.read_csv('Symptom-severity.csv')

    df.isna().sum()
    df.isnull().sum()

    cols = df.columns
    data = df[cols].values.flatten()

    s = pd.Series(data)
    s = s.str.strip()
    s = s.values.reshape(df.shape)

    df = pd.DataFrame(s, columns=df.columns)

    df = df.fillna(0)
    df = df.replace("foul_smell_of urine", 0)
    df = df.replace("dischromic _patches", 0)
    df = df.replace("spotting_ urination", 0)
    vals = df.values
    symptoms = df1['Symptom'].unique()

    for i in range(len(symptoms)):
        vals[vals == symptoms[i]] = df1[df1['Symptom'] == symptoms[i]]['weight'].values[0]
        
    d = pd.DataFrame(vals, columns=cols)
    (df[cols] == 0).all()

    df['Disease'].value_counts()
    df['Disease'].unique()
    data = df.iloc[:,1:].values
    labels = df['Disease'].values
    print(df.head())
    x_train, x_test, y_train, y_test = train_test_split(data, labels, shuffle=True, train_size = 0.85)
    #print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    scores = model.score(x_test, y_test)
    print(scores)
    preds = model.predict(x_test)
    print(preds)

def combined_sum(sum_dict):
    counter = 0
    disease_sum = {}
    with open('dataset.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader: 
            name = row[0]
            if "Disease" in name:
                continue 
            tmpSum = 0
            symptom_vec = []
            for symp in row[1:]:
                if symp == "":
                    continue

                sym_name = symp.strip().replace(" ", "") 
                symptom_vec.append(sym_name)
                val = sum_dict[sym_name]
                tmpSum += val
            if name not in disease_sum:
                disease_sum[name]=[[tmpSum],symptom_vec]
            else:
                tmplist = disease_sum[name]
                for sub in symptom_vec:
                    if sub not in tmplist[1]:
                        tmplist[1].append(sub)
                    else:
                        continue

                if tmpSum not in tmplist[0]:
                    tmplist[0].append(tmpSum)
                    #disease_sum[name]=tmplist
                else:
                    continue 
    return disease_sum




def plot_data(range_set, names):
    #    ranges = [(2, 9), (4, 10)]
    ranges = range_set
    # Set up the plot
    fig, ax = plt.subplots()
    # Plot each range as a separate line above the previous one
    for i, (start, end) in enumerate(ranges):
        ax.plot([start, end], [i, i], marker='|', markersize=12, linewidth=2, label=f'{names[i]}')
    # Add labels and title
    ax.set_yticks(range(len(ranges)))
    ax.set_yticklabels([f'{names[i]}' for i in range(len(ranges))])
    ax.set_xlabel('Value')
    ax.set_title('Ranges on Number Line')
    # Add a legend
    ax.legend()
    # Show the plot
    plt.show()

def main():
    severity_dic ={}
    with open('Symptom-severity.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            name, weight = row[0].split(",")
            if weight == "weight":
                continue 
            severity_dic[name] = int(weight)

    total_sum_disease = combined_sum(severity_dic)
    range_sets = []
    name_sets = []
    for key in total_sum_disease:
        tmp = total_sum_disease[key]
        print(f"{key}: {sorted(tmp[0])}    Difference: {max(tmp[0])-min(tmp[0])}") 
        tmp_range = (min(tmp[0]), max(tmp[0]))
        range_sets.append(tmp_range)
        name_sets.append(key)
#        print(f"Difference: {max(tmp[0])-min(tmp[0])}")
        print(f"All possible Symptoms for {key}: {tmp[1]}")
        most_severe = ""
        most_severe_weight = 0
        for sym in tmp[1]:
            if severity_dic[sym] > most_severe_weight:
                most_severe = sym
                most_severe_weight = severity_dic[sym]
            print(f"{sym}: {severity_dic[sym]}")
        print(f"Most Severe Symptom: {most_severe} {most_severe_weight}")
        print("\n")
        print(range_sets)
        #plot_data(range_sets, name_sets)
        train_model()

if __name__ == "__main__":
    main()
