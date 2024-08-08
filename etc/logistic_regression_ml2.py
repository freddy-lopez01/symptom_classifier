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
from sklearn.preprocessing import label_binarize


def roc_curve_gen(y_test, y_pred_proba):
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

    # Compute ROC curve and ROC area for each class
    n_classes = y_test_bin.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    print(n_classes)
    
    for i in range(n_classes):

        fpr[i], tpr[i], _ = metrics.roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    
    # Plot all ROC curves
    plt.figure()
    colors = ['aqua', 'darkorange', 'cornflowerblue']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for Multiclass')
    plt.legend(loc="lower right")
    plt.show()


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
    df = pd.read_csv("dataset.csv")
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
    df, df1 = read_data() 

    cols = df.columns
    data = df[cols].values.flatten()
    print(f"Shape: {data.shape}")

    df = df.map(lambda s: s.strip().replace(" ", "") if isinstance(s, str) else s)
    print(df)


    vals = df.values
    symptoms = df1['Symptom'].unique()
    
    for i in range(len(symptoms)):
        vals[vals == symptoms[i]] = df1[df1['Symptom'] == symptoms[i]]['weight'].values[0]
        
    print(df)
    #df.to_csv("temp_result_dataframe.csv")
    (df[cols] == 0).all()

    df['Disease'].value_counts()
    
    df['Disease'].unique()
    
    data = df.iloc[:,1:].values
    labels = df['Disease'].values
    X_train, X_test, y_train, y_test = train_test_split(data, labels, shuffle=True, test_size = 0.2)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    lr_model = LogisticRegression(solver='saga', max_iter=2500)
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    
    report = classification_report(y_test, y_pred)
    
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy : {acc*100:.4f} %\n")
    print("Classification report: \n")
    print(report)
    clf = LogisticRegression(multi_class='ovr', solver='liblinear')
    clf.fit(X_train, y_train)

    # Get the predicted probabilities for the test set
    y_pred_proba = clf.predict_proba(X_test)

    roc_curve_gen(y_test, y_pred_proba)
    #confusion_matrix(cnf_matrix)



    

if __name__ == "__main__":
    main()
