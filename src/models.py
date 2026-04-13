from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

def train_logistic(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

def train_svm(X, y):
    model = LinearSVC()
    model.fit(X, y)
    return model

def train_knn(X, y,k = 5):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X, y)
    return model

def train_kmeans(X, k=2):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X)
    return model