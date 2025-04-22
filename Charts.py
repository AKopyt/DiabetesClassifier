import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_test_selected = X_test[['BMI', 'DiabetesPedigreeFunction']]

model_base = DecisionTreeClassifier(random_state=42)
model_base.fit(X_train, y_train)
y_test_pred_base = model_base.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(
    X_test_selected['BMI'],
    X_test_selected['DiabetesPedigreeFunction'],
    c=y_test_pred_base,
    cmap='winter',
    edgecolor='k',
    alpha=0.7
)
plt.title("Bazowy model (domyślne parametry)")
plt.xlabel("BMI")
plt.ylabel("DiabetesPedigreeFunction")
plt.colorbar(label="Predykcja (0 = brak cukrzycy, 1 = cukrzyca)")
plt.grid(True)
plt.show()

model_3 = DecisionTreeClassifier(max_depth=5, min_samples_split=4, min_samples_leaf=2, random_state=42)
model_3.fit(X_train, y_train)
y_test_pred_3 = model_3.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(
    X_test_selected['BMI'],
    X_test_selected['DiabetesPedigreeFunction'],
    c=y_test_pred_3,
    cmap='winter',
    edgecolor='k',
    alpha=0.7
)
plt.title("Model 3 (max_depth=5, min_samples_split=4, min_samples_leaf=2)")
plt.xlabel("BMI")
plt.ylabel("DiabetesPedigreeFunction")
plt.colorbar(label="Predykcja (0 = brak cukrzycy, 1 = cukrzyca)")
plt.grid(True)
plt.show()