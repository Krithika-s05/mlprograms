import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


data = pd.read_csv(r"C:\Users\Admin\Desktop\ml\iris.csv")


X = data.drop("species", axis=1)
y = data["species"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = SVC(kernel="linear")


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("CORRECT PREDICTIONS:")
print("--------------------")

correct = 0
wrong = 0

for actual, predicted in zip(y_test, y_pred):
    if actual == predicted:
        print("Actual:", actual, " Predicted:", predicted)
        correct += 1

print("\nWRONG PREDICTIONS:")
print("------------------")

for actual, predicted in zip(y_test, y_pred):
    if actual != predicted:
        print("Actual:", actual, " Predicted:", predicted)
        wrong += 1

print("\nTotal Correct Predictions:", correct)
print("Total Wrong Predictions:", wrong)
