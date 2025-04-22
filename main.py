import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

configs = [
    {"max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1}, #baza
    {"max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1},
    {"max_depth": 5, "min_samples_split": 4, "min_samples_leaf": 2},
    {"max_depth": 7, "min_samples_split": 10, "min_samples_leaf": 5},
    {"max_depth": 10, "min_samples_split": 20, "min_samples_leaf": 10},
]

for i, params in enumerate(configs, start=1):
    print(f"\n Model {i} - Parametry: {params}")

    model = DecisionTreeClassifier(**params, random_state=42)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    train_precision = precision_score(y_train, y_train_pred)
    test_precision = precision_score(y_test, y_test_pred)

    train_recall = recall_score(y_train, y_train_pred)
    test_recall = recall_score(y_test, y_test_pred)

    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    print(f"Train Accuracy: {train_accuracy * 100:.2f}% | Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Train Precision: {train_precision * 100:.2f}% | Test Precision: {test_precision * 100:.2f}%")
    print(f"Train Recall: {train_recall * 100:.2f}% | Test Recall: {test_recall * 100:.2f}%")
    print(f"Train F1: {train_f1 * 100:.2f}% | Test F1: {test_f1 * 100:.2f}%")
