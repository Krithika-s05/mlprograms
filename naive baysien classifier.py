import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


data = pd.read_csv(r"C:\Users\Admin\Desktop\ml\trainingds.csv")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Actual Values:", list(y_test))
print("Predicted Values:", list(y_pred))
print("Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")
