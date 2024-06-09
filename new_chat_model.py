import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def load_symptom_weights(file_path):
    severity_dic = {}
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(reader)  # Skip the header row
        for row in reader:
            name, weight = row
            severity_dic[name.strip().replace(" ", "")] = int(weight)
    return severity_dic

def train_model():
    # Load the dataset
    df = pd.read_csv('reformated_dataset.csv')

    # Fill missing values with 0
    df = df.fillna(0)

    # Calculate the total severity score
    symptom_columns = df.columns[1:]  # Exclude the 'Disease' column
    df['Total_Severity_Score'] = df[symptom_columns].sum(axis=1)

    # Prepare the feature matrix (X) and target vector (y)
    X = df.drop('Disease', axis=1)
    y = df['Disease']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the decision tree classifier
    clf = DecisionTreeClassifier(random_state=42)

    # Train the model
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    # Detailed classification report
    print(classification_report(y_test, y_pred))

    # Generate the visualization of the decision tree
    plt.figure(figsize=(20, 10))
    tree_plot = plot_tree(clf, filled=True, feature_names=X_train.columns, class_names=clf.classes_)
    for item in tree_plot:
        if isinstance(item, plt.Text):
            continue  # Skip text elements
        if hasattr(item, 'get_bbox_patch'):
            bbox = item.get_bbox_patch()
            bbox.set_linewidth(0.3)  # Set the border thickness here

    plt.savefig('decision_tree.png', dpi=600, bbox_inches='tight')  # Save the visualization as an image
    plt.show()

    return clf, symptom_columns


def predict_disease(clf, symptom_columns, symptoms, severity_dic, disease_dict):
    input_data = pd.DataFrame(columns=symptom_columns)
    input_row = {symptom: 0 for symptom in symptom_columns}

    for symptom in symptoms:
        if symptom in severity_dic:
            input_row[symptom] = severity_dic[symptom]
    input_row['Total_Severity_Score'] = sum(input_row.values())

    input_data = input_data._append(input_row, ignore_index=True)
    predicted_disease = clf.predict(input_data)[0]
    total_severity_score = input_row['Total_Severity_Score']
    disease_info = disease_dict.get(predicted_disease, None)
    
    if disease_info:
        min_score, max_score = disease_info['sum_range']
        if min_score <= total_severity_score <= max_score:
            return predicted_disease
        else:
            return "No matching disease found within score range"
    else:
        return "Disease not found in dictionary"

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
    ranges = range_set
    fig, ax = plt.subplots()

    for i, (start, end) in enumerate(ranges):
        ax.plot([start, end], [i, i], marker='|', markersize=12, linewidth=2, label=f'{names[i]}')
    ax.set_yticks(range(len(ranges)))
    ax.set_yticklabels([f'{names[i]}' for i in range(len(ranges))])
    ax.set_xlabel('Value')
    ax.set_title('Ranges on Number Line')
    ax.legend()
    plt.show()

def sym_range_compute():
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
    res_dic = {}
    for key in total_sum_disease:
        tmp = total_sum_disease[key]
        print(f"{key}: {sorted(tmp[0])}    Difference: {max(tmp[0])-min(tmp[0])}")
        tmp_range = (min(tmp[0]), max(tmp[0]))
        range_sets.append(tmp_range)
        name_sets.append(key)
        print(f"All possible Symptoms for {key}: {tmp[1]}")
        res_dic[key]= {
            "symptoms":tmp[1],
            "sum_range":(tmp_range)
        }
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
    return res_dic


if __name__ == "__main__":
    severity_dic = load_symptom_weights('Symptom-severity.csv')
    disease_dict = sym_range_compute()
    clf, symptom_columns = train_model()

    # Example manual test
    test_symptoms = [ 'vomiting','sunken_eyes','dehydration']  # List of symptoms without weights
    
    predicted_disease = predict_disease(clf, symptom_columns, test_symptoms, severity_dic, disease_dict)
    if predicted_disease == "Gastroenteritis":
        print("---Model correctly predicted the Disease---")
    print(f'Predicted Disease: {predicted_disease}')

