import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline

def load_and_preprocess_data(filepath):
    df_train = pd.read_csv(filepath)
    df_train = df_train.fillna(0)
    label_encoder = LabelEncoder()
    df_train["Disease"] = label_encoder.fit_transform(df_train["Disease"])
    return df_train, label_encoder

def perform_grid_search(X_train, y_train, log_reg_param, k):
    selector = SelectKBest(score_func=chi2, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    joblib.dump(selector, 'selector.pkl')  # Save the selector
    
    pipeline = Pipeline([
        ('log_reg', LogisticRegression(max_iter=200))
    ])
    
    grid_search = GridSearchCV(pipeline, log_reg_param, cv=5, scoring="accuracy")
    grid_search.fit(X_train_selected, y_train)
    return grid_search

def evaluate(train_data, kmax, algo, k):
    test_scores = {"accuracy": {}, "f1": {}}
    train_scores = {"accuracy": {}, "f1": {}}
    print("Evaluating model with k-fold cross-validation")

    selector = joblib.load('selector.pkl')

    for i in range(2, kmax, 2):
        kf = KFold(n_splits=i)
        sum_train_acc = sum_test_acc = 0
        sum_train_f1 = sum_test_f1 = 0
        data = train_data

        for train_index, test_index in kf.split(data):
            train_data = data.iloc[train_index, :]
            test_data = data.iloc[test_index, :]
            X_train = train_data.drop(["Disease"], axis=1)
            y_train = train_data['Disease']
            X_test = test_data.drop(["Disease"], axis=1)
            y_test = test_data["Disease"]

            X_train_selected = selector.transform(X_train)
            X_test_selected = selector.transform(X_test)

            algo_model = algo.fit(X_train_selected, y_train)
            y_pred_train = algo_model.predict(X_train_selected)
            y_pred_test = algo_model.predict(X_test_selected)

            sum_train_acc += accuracy_score(y_train, y_pred_train)
            sum_test_acc += accuracy_score(y_test, y_pred_test)
            sum_train_f1 += f1_score(y_train, y_pred_train, average='macro', zero_division=0)
            sum_test_f1 += f1_score(y_test, y_pred_test, average='macro', zero_division=0)

        train_scores["accuracy"][i] = sum_train_acc / i
        test_scores["accuracy"][i] = sum_test_acc / i
        train_scores["f1"][i] = sum_train_f1 / i
        test_scores["f1"][i] = sum_test_f1 / i

        print(f"k-value: {i}, Average Train Accuracy: {train_scores['accuracy'][i]:.2f}, Average Test Accuracy: {test_scores['accuracy'][i]:.2f}")

    return train_scores, test_scores

def plot_combined_metrics(train_scores, test_scores, model_name):
    metrics = ["accuracy", "f1"]
    plt.figure(figsize=(12, 8))

    for metric in metrics:
        if metric in train_scores and metric in test_scores:
            train_values = list(train_scores[metric].values())
            test_values = list(test_scores[metric].values())
            k_values = list(train_scores[metric].keys())

            plt.plot(k_values, train_values, label=f"Train {metric.capitalize()}", marker='o', linestyle='-', linewidth=2)
            plt.plot(k_values, test_values, label=f"Test {metric.capitalize()}", marker='x', linestyle='--', linewidth=2)

    plt.title(f"{model_name} - K-Fold Cross-Validation Scores", fontsize=16)
    plt.xlabel("k-value", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(k_values, fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig("lr_kfold_crossval_score.png")
    plt.show()

def predict_symptoms(model, symptoms, all_symptoms, label_encoder, threshold=0.1):
    selector = joblib.load('selector.pkl')  # Load the saved selector
    
    input_data = np.zeros(len(all_symptoms))
    for symptom in symptoms:
        if symptom in all_symptoms:
            input_data[all_symptoms.index(symptom)] = 1

    # input_data has the same feature names as used during training
    input_data_df = pd.DataFrame([input_data], columns=all_symptoms)
    
    # transform the input data using the saved selector
    input_data_transformed = selector.transform(input_data_df)
    probabilities = model.predict_proba(input_data_transformed)[0]
    predicted_class = label_encoder.inverse_transform([np.argmax(probabilities)])[0]
    probabilities = {label_encoder.inverse_transform([i])[0]: prob for i, prob in enumerate(probabilities)}

    #print("All disease probabilities:")
    #for disease, probability in probabilities.items():
     #   print(f"{disease}: {probability * 100:.2f}%")

    high_probabilities = {disease: prob for disease, prob in probabilities.items() if prob >= threshold}

    print(f"\nPredicted Disease: {predicted_class} ({probabilities[predicted_class] * 100:.2f}%)")
    if high_probabilities:
        print("\nOther high probability diseases:")
        for disease, probability in high_probabilities.items():
            if disease != predicted_class:
                print(f"{disease}: {probability * 100:.2f}%")

    return predicted_class, high_probabilities

def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2), len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2), len(grid_param_1))

    _, ax = plt.subplots(1, 1, figsize=(12, 8))

    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx, :], '-o', label=name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig("lr_grid_search_score.png")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig("lr_confusion_matrix.png")
    plt.show()

def main():
    df_train, label_encoder = load_and_preprocess_data("transformed_dataset.csv")
    
    X = df_train.drop(["Disease"], axis=1)
    y = df_train["Disease"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    k = 84  # number of features to select
    log_reg_param = {
        "log_reg__C": [0.01, 0.1, 1, 10, 100],
        "log_reg__penalty": ["l2"],
        "log_reg__solver": ["liblinear"]
    }
    # perform grid search
    log_reg_grid_search = perform_grid_search(X_train, y_train, log_reg_param, k)

    print("==== Logistic Regression ====")
    print(f"Best parameters: {log_reg_grid_search.best_params_}")
    print(f"Best cross-validation score: {log_reg_grid_search.best_score_:.4f}")

    best_log_reg = log_reg_grid_search.best_estimator_

    model_dict = {
        "Logistic Regression": best_log_reg,
    }

    max_kfold = 11
    for model_name, model in model_dict.items():
        print(f"Evaluating {model_name}")
        train_scores, test_scores = evaluate(df_train, max_kfold, model.named_steps['log_reg'], k)
        print(f"train_scores: {train_scores}, test_scores: {test_scores}")
        plot_combined_metrics(train_scores, test_scores, model_name)

    input_symptoms = ['high_fever', 'red_sore_around_nose', 'yellow_crust_ooze', "blister"]
    #input_symptoms = ["continuous_sneezing","chills","fatigue","cough","high_fever","headache","swelled_lymph_nodes","malaise","phlegm", "throat_irritation", "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion","chest_pain", "loss_of_smell", "muscle_pain"]
    all_symptoms = df_train.columns.drop(["Disease"]).tolist()
    log_reg = model_dict["Logistic Regression"]

    print("Logistic Regression Prediction:")
    predict_symptoms(log_reg, input_symptoms, all_symptoms, label_encoder)

    # evaluate on the hold-out test set
    selector = joblib.load('selector.pkl')
    X_test_selected = selector.transform(X_test)
    y_test_pred = best_log_reg.predict(X_test_selected)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
    print(f"Hold-out Test Set Accuracy: {test_accuracy:.4f}")
    print(f"Hold-out Test Set F1 Score: {test_f1:.4f}")

    # plot grid search results for Logistic Regression
    plot_grid_search(log_reg_grid_search.cv_results_, log_reg_param['log_reg__C'], log_reg_param['log_reg__penalty'], 'C', 'Penalty')

    # plot confusion matrix
    disease_labels = label_encoder.inverse_transform(np.unique(y_test))
    plot_confusion_matrix(y_test, y_test_pred, disease_labels)

if __name__ == "__main__":
    main()
