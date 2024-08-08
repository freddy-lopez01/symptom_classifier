import pprint
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from subprocess import call
import pandas as pd 




def read_data():
    data = pd.read_csv("dataset.csv")
    print()data.head()

def main():
    data = read_data() 

if __name__ == "__main__":
    main()
