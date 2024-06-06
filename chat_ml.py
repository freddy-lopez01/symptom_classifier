import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from sklearn.utils import shuffle

def main():
    # Load the dataset
    df = pd.read_csv("dataset.csv")
    df = shuffle(df, random_state=30)
    print("First five rows of the dataset:")
    print(df.head())
    print("\nDescription of the dataset:")
    print(df.describe())
    
    # Trim whitespace from the entire dataframe
    df = df.applymap(lambda x: x.strip().replace(" ", "") if isinstance(x, str) else x)
    
    # Fill any missing values with 0
    df = df.fillna(0)
    print("\nFirst five rows after filling missing values:")
    print(df.head())
    
    null_checker = df.apply(lambda x: sum(x.isnull())).to_frame(name='count')
    print("\nNull value check:")
    print(null_checker)
    
    # Load the symptom severity data
    df1 = pd.read_csv('Symptom-severity.csv')
    df1['Symptom'] = df1['Symptom'].str.strip()  # Ensure no leading/trailing spaces
    print("\nFirst five rows of Symptom-severity data:")
    print(df1.head())

    symptom_to_weight = dict(zip(df1['Symptom'], df1['weight']))

    
    # Replace symptom names with their severity weights
    vals = df.values
    symptoms = df1['Symptom'].unique()
    for i in range(len(symptoms)):
        vals[vals == symptoms[i]] = df1[df1['Symptom'] == symptoms[i]]['weight'].values[0]
    
    d = pd.DataFrame(vals, columns=df.columns)
    #d = d.replace('dischromic  patches', 0)
    #d = d.replace('spotting  urination', 0)
    #df = d.replace('foul smell of urine', 0)
    print("\nFirst ten rows after replacing symptom names with weights:")
    print(df.head(10))
    
    null_checker = df.apply(lambda x: sum(x.isnull())).to_frame(name='count')
    print("\nNull value check after replacement:")
    print(null_checker)
    
    print("\nNumber of symptoms used to identify the disease:", len(df1['Symptom'].unique()))
    print("Number of diseases that can be identified:", len(df['Disease'].unique()))
    print("Unique diseases:", df['Disease'].unique())
    
    # Split the data into features and labels
    data = df.iloc[:, 1:].values
    labels = df['Disease'].values
    x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size=0.8, random_state=42)
    print("\nShapes of train and test sets:")
    print("x_train:", x_train.shape, "x_test:", x_test.shape, "y_train:", y_train.shape, "y_test:", y_test.shape)
    
    # Train a Decision Tree Classifier
    tree = DecisionTreeClassifier(criterion='gini', random_state=42, max_depth=13)

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    intermediate_accuracies = []

    for train_idx, val_idx in kfold.split(x_train):
        x_ktrain, x_kval = x_train[train_idx], x_train[val_idx]
        y_ktrain, y_kval = y_train[train_idx], y_train[val_idx]
        
        # Fit the model
        tree.fit(x_ktrain, y_ktrain)
        
        # Predict and calculate accuracy on the validation set
        val_preds = tree.predict(x_kval)
        accuracy = accuracy_score(y_kval, val_preds)
        intermediate_accuracies.append(accuracy)

    # Print and plot intermediate accuracies
    print("\nIntermediate Accuracies during Training:")
    for i, acc in enumerate(intermediate_accuracies):
        print(f"Fold {i+1}: {acc*100:.2f}%")

    plt.plot(range(1, len(intermediate_accuracies) + 1), intermediate_accuracies, marker='o', linestyle='-')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('Intermediate Accuracies during Training')
    plt.grid(True)
    plt.show()
    #tree.fit(x_train, y_train)
    preds = tree.predict(x_test)
    
    # Compute and plot the confusion matrix
    conf_mat = confusion_matrix(y_test, preds)
    df_cm = pd.DataFrame(conf_mat, index=np.unique(labels), columns=np.unique(labels))
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Cross-validation on the training set
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    DS_train = cross_val_score(tree, x_train, y_train, cv=kfold, scoring='accuracy')
    print("\nCross-validation on training set:")
    print("Mean Accuracy: %.3f%%, Standard Deviation: (%.2f%%)" % (DS_train.mean() * 100.0, DS_train.std() * 100.0))
    
    # Cross-validation on the test set
    DS_test = cross_val_score(tree, x_test, y_test, cv=kfold, scoring='accuracy')
    print("\nCross-validation on test set:")
    print("Mean Accuracy: %.3f%%, Standard Deviation: (%.2f%%)" % (DS_test.mean() * 100.0, DS_test.std() * 100.0))
    
    def manual_test(symptoms):
        # Convert symptoms to weights
        symptoms_weights = [symptom_to_weight.get(symptom.strip(), 0) for symptom in symptoms]
        # Pad the list to match the number of symptoms used in training
        while len(symptoms_weights) < data.shape[1]:
            symptoms_weights.append(0)
        # Convert to numpy array and reshape for prediction
        symptoms_weights = np.array(symptoms_weights).reshape(1, -1)
        # Predict the disease
        prediction = tree.predict(symptoms_weights)
        print(symptoms_weights)
        return prediction[0]



    example_symptoms = [['skin_rash', 'joint_pain', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails'], 
                        ['cough']]
    #example_symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'dischromic_patches']
    #likelihoods = manual_test3(example_symptoms)
    #print("\nLikelihood of each disease based on symptoms:")
    #for disease, likelihood in likelihoods.items():
    #    print(f"{disease}: {likelihood:.2f}%")
    
    # Example manual test
    #example_symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'dischromic_patches']
    #example_symptoms = ['stomach_bleeding', 'coma', 'blood_in_sputum', 'pain_in_anal_region']
    for i in range(len(example_symptoms)):
        predicted_disease = manual_test(example_symptoms[i])
        print(f"\nPredicted Disease for symptoms {example_symptoms[i]}: {predicted_disease}")

if __name__ == "__main__":
    main()

