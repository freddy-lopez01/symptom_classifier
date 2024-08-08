import pprint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.linear_model import LogisticRegression
from subprocess import call
import pandas as pd 


def confusion_matrix(dataframe):
    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(dataframe), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    
    plt.show()


def read_data():
    df = pd.read_csv("updated_dataset.csv")
    df = df.fillna(0)
    df1 = pd.read_csv("Symptom-severity.csv")
    
    return df, df1

def visualize_data(dataframe):
    fig = plt.figure(figsize=(20,100))

    for i, (name, row) in enumerate(dataframe.iterrows()):
        ax = plt.subplot(14,3, i+1)
        ax.set_title(row.name)
        ax.set_aspect('equal')
        for idx,val in row.items():
            if val==0:
                del row[idx] 
        ax.pie(row, labels=row.index, autopct='%.2f%%')
    
    plt.show()

def main():
    data, symptoms = read_data() 
    vis_data = data.groupby("Disease").mean()
    disease_column = data
    x = data.drop(['Disease'], axis=1)
    y = data["Disease"]
    print(x)
    print(y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=16)
    print(X_train)
    print(X_test)
    logreg = LogisticRegression(random_state=16)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    
    report = classification_report(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy : {acc*100:.4f} %\n")
    print("Classification report: \n")
    print(report)

    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    #confusion_matrix(cnf_matrix)



    

if __name__ == "__main__":
    main()
