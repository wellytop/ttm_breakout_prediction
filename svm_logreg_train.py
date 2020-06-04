import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
import numpy as np

# Load features from the disk
with (open("all_features.pickle", "rb")) as openfile:
    while True:
        try:
            all_features = pickle.load(openfile)
        except EOFError:
            break


# # Transform data for model

svm_input_X = []
svm_input_Y = []

for ticker in all_features.keys():
    print(ticker)
    for feature in all_features[ticker].keys():
        print(feature)
        stock_df = all_features[ticker][feature]["data"]
        label = all_features[ticker][feature]["label"]
        ind = stock_df.index.get_loc(0)
        stock_target_df = stock_df.iloc[
            :ind
        ]  # last day of the set minus 120 days and plus 21 days
        time_series = stock_target_df["close"].values.reshape(1, -1)

        if label != "ambiguous" and len(time_series[0]) == 120:
            svm_input_X.append(time_series[0])
            svm_input_Y.append(label)
            break


# Split Data to training and testing
X_train, X_test, y_train, y_test = train_test_split(
    svm_input_X, svm_input_Y, test_size=0.2
)


# Train Support Vector Machine (SVM) Model
tuned_parameters = [
    {"C": [1, 10, 100, 1000], "kernel": ["linear"]},
    {"C": [1, 10, 100, 1000], "gamma": [0.001, 0.0001], "kernel": ["rbf"]},
]
svm = GridSearchCV(SVC(), cv=5, param_grid=tuned_parameters, verbose=4)
svm.fit(X_train, y_train)

# Evaluate SVM Model
y_pred = svm.predict(X_test)
correct = accuracy_score(y_test, y_pred, normalize=False)
print("The accuracy is ", correct / len(y_test))

# Save SVM mode as pickle file
pickle_out = open("svm_model.pickle", "wb")
pickle.dump(svm, pickle_out)
pickle_out.close()


# Train Logistic Regression (LR) Model
tuned_parameters = [
    {"penalty": ["l1", "l2"], "C": np.logspace(-4, 4, 20), "solver": ["liblinear"]}
]
logisticRegr = GridSearchCV(
    LogisticRegression(), cv=5, param_grid=tuned_parameters, verbose=4
)
logisticRegr.fit(X_train, y_train)


# Evaluate LR Model
y_pred = logisticRegr.predict(X_test)
correct = accuracy_score(y_test, y_pred, normalize=False)
print("The accuracy is ", correct / len(y_test))

# Save LR mode as pickle file
pickle_out = open("logreg_model.pickle", "wb")
pickle.dump(logisticRegr, pickle_out)
pickle_out.close()
