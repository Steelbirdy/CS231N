import numpy as np
import timer

X = np.array([[1, 2], [3, 4]])
X_train = np.array([[7, 8], [9, 10]])
dists = np.zeros((X.shape[0], X_train.shape[0]))


for i in range(X.shape[0]):
    for j in range(X_train.shape[0]):
        dists[i, j] = np.sqrt(np.sum(np.square(X_train[j] - X[i])))
print(dists)

for i in range(X.shape[0]):
    dists[i, :] = np.sqrt(np.sum(np.square(X_train - X[i, :]), axis=1))
print(dists)

dists = np.sqrt(np.sum(X * X, axis=1, keepdims=True) + np.sum(X_train * X_train, axis=1) - 2 * np.dot(X, X_train.T))
print(dists)
