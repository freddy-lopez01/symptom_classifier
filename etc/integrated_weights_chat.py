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
        next(reader)
        for row in reader:
            name, weight = row
            severity_dic[name.strip().replace(" ", "")] = int(weight)
    return severity_dic

def combined_sum(sum_dict):
    disease_sum = {}
    with open('dataset.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
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
                disease_sum[name] = [[tmpSum], symptom_vec]
            else:
                tmplist = disease_sum[name]
                for sub in symptom_vec:
                    if sub not in tmplist[1]:
                        tmplist[1].append(sub)
                if tmpSum not in tmplist[0]:
                    tmplist[0].append(tmpSum)
    return disease_sum

def sym_range_compute():
    severity_dic = load_symptom_weights('Symptom-severity.csv')
    total_sum_disease = combined_sum(severity_dic)
    range_sets = []
    name_sets = []
    res_dic = {}
    for key in total_sum_disease:
        tmp = total_sum_disease[key]
        tmp_range = (min(tmp[0]), max(tmp[0]))
        range_sets.append(tmp_range)
        name_sets.append(key)
        res_dic[key] = {
            "symptoms": tmp[1],
            "sum_range": tmp_range
        }
    return res_dic

def train_model():
    df = pd.read_csv('reformated_dataset.csv')
    df = df.fillna(0)
    symptom_columns = df.columns[1:]
    df['Total_Severity_Score'] = df[symptom_columns].sum(axis=1)

    X = df.drop('Disease', axis=1)
    y = df['Disease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    print(classification_report(y_test, y_pred))
    plt.figure(figsize=(20, 10))
    tree_plot = plot_tree(clf, filled=True, feature_names=X_train.columns, class_names=clf.classes_)
    for item in tree_plot:
        if isinstance(item, plt.Text):
            continue
        if hasattr(item, 'get_bbox_patch'):
            bbox = item.get_bbox_patch()
            bbox.set_linewidth(0.3)

    plt.savefig('decision_tree_weights.png', dpi=600, bbox_inches='tight')
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

def get_disease_info(disease_name):
    disease_info = pd.read_csv("symptom_precaution.csv")
    disease_info.columns = disease_info.columns.str.strip()
    try:
        result = disease_info.loc[disease_info['Disease'] == disease_name]
        if result.empty:
            return f"No information found for disease: {disease_name}"
        else:
            print(f"Based on your symptoms, the model has predicted a diagnosis of {disease_name}")
            print("Here are a few recommendations to help counteract and reduce such symptoms:")
            for index, row in result.iterrows():
                for column in result.columns[1:]:
                    print(f"{column}: {row[column]}")
            return None
    except KeyError:
        return f"Disease column not found in the DataFrame"

if __name__ == "__main__":
    severity_dic = load_symptom_weights('Symptom-severity.csv')
    disease_dict = sym_range_compute()
    clf, symptom_columns = train_model()

    test_symptoms = ['vomiting', 'sunken_eyes', 'dehydration']
    predicted_disease = predict_disease(clf, symptom_columns, test_symptoms, severity_dic, disease_dict)
    get_disease_info(predicted_disease)

    if predicted_disease == "Gastroenteritis":
        print("---Model correctly predicted the Disease---")
    print(f'Predicted Disease: {predicted_disease}')

