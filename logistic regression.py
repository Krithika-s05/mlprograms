import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


train_data = pd.read_csv(r"C:\Users\Admin\Desktop\ml\trainingds.csv")

X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)


test_data = pd.read_csv(r"C:\Users\Admin\Desktop\ml\testingds.csv")

X1 = test_data.iloc[:, :-1]
y1 = test_data.iloc[:, -1]


X1_scaled = scaler.transform(X1)


y_pred = model.predict(X1_scaled)


accuracy = accuracy_score(y1, y_pred)


print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
print("Actual Values:", list(y1))
print("Predicted Values:", list(y_pred))
print("Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")
