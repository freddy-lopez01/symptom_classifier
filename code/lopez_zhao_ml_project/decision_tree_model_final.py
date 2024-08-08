import pandas as pd
import seaborn as sns
import textwrap
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def load_symptom_weights(file_path):
    """
    load_symptom_weights() takes in a file_path and loads the spcified file and reads in the data and stores it as a dictionary. 
    It then modifies/corrects the data in order to prevent any inconsistencies in the data
    returns a dictionary
    """
    severity_dic = {}
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(reader)
        for row in reader:
            name, weight = row
            severity_dic[name.strip().replace(" ", "")] = int(weight)
    return severity_dic
def train_multiple(X_train, y_train, min_samples_split):
    """
    train_multiple() takes in three parameters: X_train, y_train, min_samples_splits. It then applies each of the parameters to the change the hyperparameters of the tree prior to training. 
    returns the trained tree
    """
    clf = DecisionTreeClassifier(random_state=42, min_samples_split=min_samples_split, max_depth=50)
    clf.fit(X_train, y_train)
    return clf

def train_model():
    """
    train_model() reads in data from 'reformated_dataset.csv' and converts the data in to a pandas dataframe. It then trains the tree with different 
    hyper parameter settings and returns the settings that yield the highest accuracy
    Finally, it creates a classification report based off of the metrics computed by the classification_report() method from sklearn.metrics
    """
    df = pd.read_csv('../dataset/reformated_dataset.csv')
    df = df.fillna(0)
   # df_head = df.iloc[:5, :6]

   # fig, ax = plt.subplots(figsize=(20, 10))  # Adjust the size as needed

   # ax.xaxis.set_visible(False)
   # ax.yaxis.set_visible(False)
   # ax.set_frame_on(False)
   # 
   # tbl = ax.table(cellText=df_head.values, colLabels=df_head.columns, cellLoc='center', loc='center')
   # 
   # tbl.auto_set_font_size(False)
   # tbl.set_fontsize(12)
   # tbl.scale(1.2, 1.2)
   # 
   # plt.savefig('df_updated_head.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
   # plt.close()
   # 
   # print("DataFrame head saved as df_head.png")


    symptom_columns = df.columns[1:]
    df['Total_Severity_Score'] = df[symptom_columns].sum(axis=1)

    X = df.drop('Disease', axis=1)
    y = df['Disease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    min_samples_splits = range(2, 100)
    accuracies = []

    for min_samples_split in min_samples_splits:
        clf = train_multiple(X_train, y_train, min_samples_split)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    plt.figure(figsize=(10, 6))
    plt.plot(min_samples_splits, accuracies, marker='o')
    plt.title('Accuracy of Decision Tree with Different min_samples_split')
    plt.xlabel('min_samples_split')
    plt.ylabel('Accuracy')
    plt.grid(True)

    best_min_samples_split = min_samples_splits[np.argmax(accuracies)]
    print(f'Best min_samples_split: {best_min_samples_split} with accuracy: {max(accuracies):.2f}')
    
    # Train the final model with the best parameter and evaluate it
    clf = train_multiple(X_train, y_train, best_min_samples_split)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {accuracy:.2f}')
    print(classification_report(y_test, y_pred))
    plt.figure(figsize=(20, 10))
    tree_plot = plot_tree(clf, filled=True, feature_names=X_train.columns, class_names=clf.classes_)
    for item in tree_plot:
        if isinstance(item, plt.Text):
            continue  # Skip text elements
        if hasattr(item, 'get_bbox_patch'):
            bbox = item.get_bbox_patch()
            bbox.set_linewidth(0.3)  

    plt.savefig('decision_tree.png', dpi=600, bbox_inches='tight') 
    plt.show()

    plot_classification_report(y_test, y_pred)

    return clf, symptom_columns

def plot_classification_report(y_true, y_pred):
    """
    plot_classification_report() plots the report data created by the sklearn classification_report() method. 
    it then shows the plotted data that later can be saved to my local machine for use later in the report and for metric comparison 
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    # Drop support column and average rows
    df_report.drop(columns=['support'], inplace=True)
    df_report.drop(index=['accuracy', 'macro avg', 'weighted avg'], inplace=True)

    plt.figure(figsize=(20, 10))
    sns.heatmap(df_report, annot=True, cmap='Reds', fmt='.2f')
    plt.title('Classification Report')
    plt.ylabel('Classes')
    plt.xlabel('Metrics')
    plt.show()


def predict_disease(clf, symptom_columns, symptoms, severity_dic, disease_dict):
    """
    predicted_disease() takes in the trained model,
    """
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
    with open('../dataset/dataset.csv', newline='') as csvfile:
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
    with open('../dataset/Symptom-severity.csv', newline='') as csvfile:
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
        #print(f"{key}: {sorted(tmp[0])}    Difference: {max(tmp[0])-min(tmp[0])}")
        tmp_range = (min(tmp[0]), max(tmp[0]))
        range_sets.append(tmp_range)
        name_sets.append(key)
        #print(f"All possible Symptoms for {key}: {tmp[1]}")
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
    #        print(f"{sym}: {severity_dic[sym]}")
    #   print(f"Most Severe Symptom: {most_severe} {most_severe_weight}")
    #    print("\n")
    #    print(range_sets)
    return res_dic


def get_disease_info(disease_name):
    disease_info = pd.read_csv("symptom_precaution.csv")
    symptoms_info = pd.read_csv("symptom_Description.csv")
    disease_info.columns = disease_info.columns.str.strip()
    try:
        result = disease_info.loc[disease_info['Disease'] == disease_name]
        res_desc = symptoms_info.loc[disease_info['Disease'] == disease_name]
        if result.empty:
            return f"No information found for disease: {disease_name}"
        else:
            print(f"Based on your symptoms, the model has predictied a diagnosis of {disease_name}\n")
            description = res_desc.iloc[0]['Description']
            print_wrapped_text(description)
            print("Here are a few recommandations to help counteract and reduce such symptoms:")
            for index, row in result.iterrows():
                for column in result.columns[1:]:
                    print(f"{column}: {row[column]}")
            return None
    except KeyError:
        return f"Disease column not found in the DataFrame"

def print_wrapped_text(text, width=70):
    wrapper = textwrap.TextWrapper(width=width)
    wrapped_text = wrapper.fill(text)
    print(wrapped_text)


if __name__ == "__main__":
    severity_dic = load_symptom_weights('../dataset/Symptom-severity.csv')
    disease_dict = sym_range_compute()
    count = 0
    for key, value in disease_dict.items():
	    count += 1
    print(f"disease count: {count}")
    count1 = 0	
    for key, value in severity_dic.items():
	    count1 += 1
    print(f"disease count: {count1}")
	
#    clf, symptom_columns = train_model()

    #test_symptoms = [ 'vomiting','sunken_eyes','dehydration']
    #test_symptoms = ["vomiting", "dehydration", "diarrhoea"]
#    test_symptoms = ["continuous_sneezing","chills","fatigue","cough","high_fever","headache","swelled_lymph_nodes","malaise","phlegm", "throat_irritation", "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion","chest_pain", "loss_of_smell", "muscle_pain"]
    #test_symptoms = ['high_fever', 'red_sore_around_nose', 'yellow_crust_ooze']
    #predicted_disease = predict_disease(clf, symptom_columns, test_symptoms, severity_dic, disease_dict)
#    get_disease_info(predicted_disease)

