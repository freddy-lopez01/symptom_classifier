import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix,classification_report,confusion_matrix,precision_score,roc_curve
import seaborn as sns
from sklearn.utils import shuffle
# from pandas_profiling import ProfileReport
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier 
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import csv


def train_model():
    pass



def main():
    df = pd.read_csv("dataset.csv")
    df = shuffle(df,random_state=30)
    print(df.head())
    print(df.describe())
    cols = df.columns
    data = df[cols].values.flatten()
    s = pd.Series(data)
    s = s.values.reshape(df.shape)

    df = pd.DataFrame(s, columns=df.columns)

    df = df.fillna(0)
    df.head()
    null_checker = df.apply(lambda x: sum(x.isnull())).to_frame(name='count')
    print(null_checker)
    df1 = pd.read_csv('Symptom-severity.csv')
    df1.head()

    vals = df.values
    symptoms = df1['Symptom'].unique()

    for i in range(len(symptoms)):
        vals[vals == symptoms[i]] = df1[df1['Symptom'] == symptoms[i]]['weight'].values[0]
    
    d = pd.DataFrame(vals, columns=cols)
    d.head()
    d = d.replace('dischromic  patches', 0)
    d = d.replace('spotting  urination',0)
    df = d.replace('foul smell of urine',0)
    df.head(10)
    null_checker = df.apply(lambda x: sum(x.isnull())).to_frame(name='count')
    print(null_checker)
    print("Number of symptoms used to identify the disease ",len(df1['Symptom'].unique()))
    print("Number of diseases that can be identified ",len(df['Disease'].unique()))
    print(df['Disease'].unique())
    data = df.iloc[:,1:].values
    labels = df['Disease'].values
    x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size = 0.8,random_state=42)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    #F1 = 2 * (precision * recall) / (precision + recall)
    tree =DecisionTreeClassifier(criterion='gini',random_state=42,max_depth=13)
    tree.fit(x_train, y_train)
    preds=tree.predict(x_test)
    conf_mat = confusion_matrix(y_test, preds)
    df_cm = pd.DataFrame(conf_mat, index=df['Disease'].unique(), columns=df['Disease'].unique())
   # print('F1-score% =', f1_score(y_test, preds, average='macro')*100, '|', 'Accuracy% =', accuracy_score(y_test, preds)*100)
    sns.heatmap(df_cm)
    kfold = KFold(n_splits=10,shuffle=True,random_state=42)
    DS_train =cross_val_score(tree, x_train, y_train, cv=kfold, scoring='accuracy')
    pd.DataFrame(DS_train,columns=['Scores'])
    print("Mean Accuracy: %.3f%%, Standard Deviation: (%.2f%%)" % (DS_train.mean()*100.0, DS_train.std()*100.0))
    kfold = KFold(n_splits=10,shuffle=True,random_state=42)
    DS_test =cross_val_score(tree, x_test, y_test, cv=kfold, scoring='accuracy')
    pd.DataFrame(DS_test,columns=['Scores'])
    print("Mean Accuracy: %.3f%%, Standard Deviation: (%.2f%%)" % (DS_test.mean()*100.0, DS_test.std()*100.0))








if __name__ == "__main__":
    main()
