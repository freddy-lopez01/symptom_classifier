import pandas as pd
import pprint
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from subprocess import call

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # loads CSV files
    df = pd.read_csv("dataset.csv")
    df1 = pd.read_csv("Symptom-severity.csv")
    print(df.head())
    print(df1.head())
    return df, df1

def preprocess_data(df: pd.DataFrame, df1: pd.DataFrame) -> pd.DataFrame:
    #print(df.isna().sum(), df.isnull().sum())
    cols = df.columns
    # flatten and strip any leading/trailing whitespace
    data = df[cols].values.flatten()
    s = pd.Series(data).str.strip().values.reshape(df.shape)
    df = pd.DataFrame(s, columns=df.columns)
    # set empty elements to zero and replace incorrect symptom format
    df = df.fillna(0).replace(["foul_smell_of urine", "dischromic _patches", "spotting_ urination"], 0)
    print(f"data: {df}")

    # extract values from dataframe and get unique symptoms
    value = df.values
    symptoms = df1["Symptom"].unique()
    # replace symptoms with corresponding weights
    for i in range(len(symptoms)):
        value[value == symptoms[i]] = df1[df1["Symptom"] == symptoms[i]]["weight"].values[0]
    df = pd.DataFrame(value, columns=cols)
    print((df[cols] == 0).all())
    
    return df

def visualize_data(df: pd.DataFrame) -> None:
    # plot distribution of diseases
    plt.figure(figsize=(12, 6))
    df["Disease"].value_counts().plot(kind="bar")
    plt.title("Distribution of Diseases")
    plt.xlabel("Disease")
    plt.ylabel("Count")
    plt.show()

def train_model(data: pd.DataFrame, labels):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, shuffle=True, train_size=0.85)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    return model, x_test, y_test

def evaluate_model(model: RandomForestClassifier, x_test: pd.DataFrame, y_test: pd.DataFrame) -> None:
    scores = model.score(x_test, y_test)
    print(f"Model Accuracy: {scores}")
    preds = model.predict(x_test)
    print(f"Predictions: {preds}")

def export_tree(model: RandomForestClassifier, cols: pd.DataFrame, df: pd.DataFrame) -> None:
    estimator = model.estimators_[5]
    feature_names = cols[1:]  
    export_graphviz(estimator, out_file="tree.dot", 
                    feature_names=feature_names,
                    class_names=df["Disease"].unique(),
                    rounded=True, proportion=False, 
                    precision=2, filled=True)
    call(["dot", "-Tpng", "tree.dot", "-o", "tree.png", "-Gdpi=600"])

def main():
    df, df1 = load_data()
    df = preprocess_data(df, df1)
    print(df['Disease'].value_counts())
    print(df['Disease'].unique())
    data = df.iloc[:, 1:].values
    labels = df['Disease'].values
    print(df.head())

    visualize_data(df)
    model, x_test, y_test = train_model(data, labels)

    export_tree(model, df.columns, df)
    evaluate_model(model, x_test, y_test)
 

if __name__ == "__main__":
    main()
