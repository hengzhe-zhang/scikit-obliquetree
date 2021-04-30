from copy import deepcopy

import numpy as np


class ContinuouslyOptimizedObliqueRegressionTree:
    def __init__(
        self,
        impurity,
        segmentor,
        nu=1.0,
        thau=10,
        max_iter=100,
        tol=1e-6,
        step=0.1,
    ):
        self.impurity = impurity
        self.segmentor = segmentor
        self.nu = nu
        self.thau = thau
        self.max_iter = max_iter
        self.tol = tol
        self.step = step

    def fit(self, X, y):

        X_train = deepcopy(X)

        impurity, sr, left_indices, right_indices = self.segmentor(
            X_train, y, self.impurity
        )

        X_train = np.hstack((X_train, np.ones(X_train.shape[0])[:, np.newaxis]))
        n_objects, n_features = X_train.shape

        w_new = np.zeros(n_features)
        w_new[sr[0]] = 1.0
        w_new[-1] = -sr[1]
        theta_left = y[left_indices].mean()
        theta_right = y[right_indices].mean()
        self.l = []
        l_old = np.inf
        l_new = self.objective(w_new, theta_left, theta_right, X_train, y)
        it = 0

        while np.abs(l_old - l_new) > self.tol:
            it += 1
            w_old = w_new
            for i in range(self.thau):
                ind_batch = np.random.choice(np.arange(n_objects), size=1)
                X_batch = X_train[ind_batch, :]
                y_batch = y[ind_batch]

                s = np.sign((w_old * X_batch).sum())

                if (
                    -(w_new * X_batch).sum() + (theta_left - y_batch) ** 2
                    >= (w_new * X_batch).sum() + (theta_right - y_batch) ** 2
                ):
                    w_new = w_new + self.step * (1 + s) * X_batch
                    theta_left = theta_left - 2 * self.step * (
                        theta_left - y_batch
                    )
                else:
                    w_new = w_new - self.step * (1 - s) * X_batch
                    theta_right = theta_right - 2 * self.step * (
                        theta_right - y_batch
                    )

                if (w_new * w_new).sum() > self.nu:
                    w_new = np.sqrt(self.thau) * w_new / np.linalg.norm(w_new)

            l_old = self.objective(w_old, theta_left, theta_right, X_train, y)
            l_new = self.objective(w_new, theta_left, theta_right, X_train, y)
            self.l.append(
                self.objective(w_new, theta_left, theta_right, X_train, y)
            )
            if it >= self.max_iter:
                break

        self.weights = w_new
        self.theta_left = theta_left
        self.theta_right = theta_right

    def objective(self, w, theta_left, theta_right, X, y):
        loss = np.vstack(
            (
                -w.dot(X.T) + (theta_left - y) ** 2,
                w.dot(X.T) + (theta_right - y) ** 2,
            )
        )
        return (np.max(loss, axis=0) - np.abs(w.dot(X.T))).sum()

    def predict(self, X):

        X_test = deepcopy(X)
        X_test = np.hstack((X_test, np.ones(X_test.shape[0])[:, np.newaxis]))
        n_objects, n_features = X_test.shape
        y = np.zeros(n_objects)
        mask = (self.weights.dot(X_test.T) >= 0).flatten()
        y[mask] = self.theta_right
        y[np.logical_not(mask)] = self.theta_left

        return y
