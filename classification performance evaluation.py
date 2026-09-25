import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, roc_auc_score


data = pd.read_csv(r"C:\Users\Admin\Desktop\ml\Heart (1).csv")


data.columns = data.columns.str.strip()


data = data.dropna()


for column in data.columns:
    if data[column].dtype == 'object':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].astype(str))


X = data.drop("AHD", axis=1)
y = data["AHD"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(probability=True)
}

results = {}


for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    results[name] = [accuracy, precision, recall, f1, auc]


print("\nMODEL PERFORMANCE")
print("-----------------")

for name, values in results.items():

    print("\n", name)
    print("Accuracy  :", round(values[0], 4))
    print("Precision :", round(values[1], 4))
    print("Recall    :", round(values[2], 4))
    print("F1-Score  :", round(values[3], 4))
    print("ROC-AUC   :", round(values[4], 4))


best_model = max(results, key=lambda x: results[x][0])

print("\nBest Performing Model:", best_model)


for name, values in results.items():
    plt.scatter(
        values[0],
        values[4],
        s=100,
        label=name
    )

plt.xlabel("Accuracy")
plt.ylabel("ROC-AUC")
plt.title("Classifier Performance Comparison")
plt.legend()
plt.grid()
plt.show()
