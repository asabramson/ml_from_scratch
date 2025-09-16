import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from models.linear_regression import LinearRegression
from models.logistic_regression import LogisticRegression
from models.knn import KNNClassifier
from models.kmeans import KMeans
from models.pca import PCA
from models.naive_bayes import NaiveBayes
from models.svm import SVM
from models.decision_tree import DecisionTree
from models.random_forest import RandomForest
from utils import accuracy, mse_func, standardize

def test_linear_regression():
    X, y = datasets.make_regression(n_samples=1000, n_features=10, n_informative=8,
            noise=2.0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    reg = LinearRegression(lr=0.05)
    reg.fit(X_train,y_train)
    preds = reg.predict(X_test)

    mse = mse_func(y_test, preds)
    print(f"Linear Regression MSE: {mse:.4f}")

def test_logistic_regression():
    X, y = datasets.load_breast_cancer(return_X_y=True)

    # Fixed split (no shuffle)
    # split = int(0.8 * X.shape[0])
    # X_train, X_test = X[:split], X[split:]
    # y_train, y_test = y[:split], y[split:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression(lr=0.01, n_iters=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy(y_test, y_pred)
    print(f"Logistic Regression accuracy: {acc:.4f}")

def test_knn():
    X, y = datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train, mean, std = standardize(X_train)
    X_test, _, _ = standardize(X_test, mean, std)

    clf = KNNClassifier(k=5)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy(y_test, y_pred)
    print("KNN accuracy:", f"{acc:.4f}")

def test_kmeans():
    X, y = datasets.make_blobs(n_samples=100, centers=5, cluster_std=1.0)

    X, mean, std = standardize(X)

    km = KMeans(k=5, max_iters=200, tol=1e-4).fit(X)
    clusters = len(np.unique(y))
    print("KMeans")
    print(f"  Centers shape: {km.centers_.shape}")
    print(f"  Inertia: {round(km.inertia_, 4)}")
    print(f"  Labels (first 10): {km.labels_[:10]}")

    new_points = np.array([[1.0, 2.0], [23.0, -30.0], [2.0, -3.0]])
    new_points, _, _ = standardize(new_points, mean, std)
    cluster_ids = km.predict(new_points)

    print(f"New data clusters: {cluster_ids}")

def test_pca():
    X, y = datasets.load_breast_cancer(return_X_y=True)

    X, mean, std = standardize(X)

    pca = PCA(n_components=6).fit(X)
    Z = pca.transform(X)

    print("PCA")
    print(f"  Original Shape: {X.shape}")
    print(f"  Z shape: {Z.shape}")
    print(f"  Explained variance ratio: {np.round(pca.explained_variance_ratio_, 4)}, Sum: {round(float(pca.explained_variance_ratio_.sum()), 4)}")
    X_hat = pca.inverse_transform(Z)
    mse_rec = mse_func(X, X_hat)
    print(f"  Reconstruction MSE (how much data was lost in the dimension reduction?): {round(mse_rec, 4)}")

def test_naive_bayes():
    X, y = datasets.load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = NaiveBayes().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"Gaussian Naive Bayes accuracy: {accuracy(y_test, y_pred):.4f}")

def test_svm():
    X, y = datasets.load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train, mean, std = standardize(X_train)
    X_test, _, _ = standardize(X_test, mean, std)

    clf = SVM(lr=0.001, n_iters=1000, C=1.0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"Linear SVM accuracy: {accuracy(y_test, y_pred):.4f}")

def test_decision_tree():
    X, y = datasets.load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = DecisionTree(max_depth=10, min_samples_split=2, min_impurity_decrease=0.0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"Decision Tree accuracy: {accuracy(y_test, y_pred):.4f}")

def test_random_forest():
    X, y = datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    rf = RandomForest(
        n_trees=40,
        max_depth=None,
        min_samples_split=2,
        min_impurity_decrease=0.0
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print(f"Random forest accuracy: {accuracy(y_test, y_pred):.4f}")


if __name__ == "__main__":
    print("Uncomment tests below")
    # test_linear_regression()
    # test_logistic_regression()
    # test_knn()
    # test_kmeans()
    test_pca()
    # test_naive_bayes()
    # test_svm()
    # test_decision_tree()
    # test_random_forest()