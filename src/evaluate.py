from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return acc, f1

def evaluate_kmeans(model, X, y_true):
    y_pred = model.predict(X)
    acc1 = accuracy_score(y_true, y_pred)
    acc2 = accuracy_score(y_true, 1 - y_pred)
    return max(acc1, acc2)