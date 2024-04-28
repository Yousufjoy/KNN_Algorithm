
import numpy as np

data = np.genfromtxt('/content/iris.csv', delimiter=',')
X = data[:, 2:6]
y = data[:, 0]
indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

train_size = int(0.8 * len(X))
X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def knn(X_train, y_train, X_test, k):
    y_pred = []
    for x_test in X_test:
        distances = [euclidean_distance(x_train, x_test) for x_train in X_train]
        k_nearest_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train[i] for i in k_nearest_indices]
        y_pred.append(np.bincount(k_nearest_labels).argmax())
    return np.array(y_pred)

k = 5
y_pred = knn(X_train, y_train, X_test, k)

accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f"Accuracy (k={k}): {accuracy * 100:.2f}%")