import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv(r"C:\Users\Admin\Desktop\ml\Heart (1).csv")
print("Dataset:")
print(data.head())
data = pd.get_dummies(data, drop_first=True)
data = data.dropna()
X = data.drop("Chol", axis=1)
y = data["Chol"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
nonlinear_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

nonlinear_model.fit(X_train, y_train)
y_pred_nonlinear = nonlinear_model.predict(X_test)
def evaluate_model(name, y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n", name)
    print("MSE  :", round(mse, 4))
    print("RMSE :", round(rmse, 4))
    print("MAE  :", round(mae, 4))
    print("R2   :", round(r2, 4))

    return mse, rmse, mae, r2
linear_results = evaluate_model(
    "Linear Regression",
    y_test,
    y_pred_linear
)

nonlinear_results = evaluate_model(
    "Non-Linear Regression",
    y_test,
    y_pred_nonlinear
)
print("\nPerformance Comparison")

if nonlinear_results[3] > linear_results[3]:
    print("Non-Linear Regression performs better.")
else:
    print("Linear Regression performs better.")
plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    y_pred_linear,
    label="Linear Regression"
)

plt.scatter(
    y_test,
    y_pred_nonlinear,
    label="Non-Linear Regression"
)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.legend()
plt.grid()
plt.show()
