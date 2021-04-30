from copy import deepcopy

import numpy as np
from models.CART.HHCART import HHCARTNode
from numba import njit
from numpy.linalg import norm
from sklearn.base import BaseEstimator


@njit(cache=True)
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result


@njit(cache=True)
def np_max(array, axis):
    return np_apply_along_axis(np.max, axis, array)


@njit(cache=True)
def objective(w, theta_left, theta_right, X, y):
    loss = np.vstack(
        (
            -w.dot(X.T) + (theta_left - y) ** 2,
            w.dot(X.T) + (theta_right - y) ** 2,
        )
    )
    return (np_max(loss, axis=0) - np.abs(w.dot(X.T))).sum()


@njit(cache=True)
def optimization(
    X_train, y, left_indices, right_indices, sr, tol, thau, step, nu, max_iter
):
    n_objects, n_features = X_train.shape
    w_new = np.zeros(n_features)
    w_new[sr[0]] = 1.0
    w_new[-1] = -sr[1]
    theta_left = y[left_indices].mean()
    theta_right = y[right_indices].mean()
    l = []
    l_old = np.inf
    l_new = objective(w_new, theta_left, theta_right, X_train, y)
    it = 0
    while np.abs(l_old - l_new) > tol:
        it += 1
        w_old = w_new
        for i in range(thau):
            ind_batch = np.random.choice(np.arange(n_objects), size=1)
            X_batch = X_train[ind_batch, :]
            y_batch = y[ind_batch]

            s = np.sign((w_old * X_batch).sum())

            if (
                -(w_new * X_batch).sum() + (theta_left - y_batch) ** 2
                >= (w_new * X_batch).sum() + (theta_right - y_batch) ** 2
            ):
                w_new = w_new + (step * (1 + s) * X_batch).flatten()
                theta_left = (theta_left - 2 * step * (theta_left - y_batch))[0]
            else:
                w_new = w_new - (step * (1 - s) * X_batch).flatten()
                theta_right = (
                    theta_right - 2 * step * (theta_right - y_batch)
                )[0]

            if (w_new * w_new).sum() > nu:
                w_new = np.sqrt(thau) * w_new / np.linalg.norm(w_new)

        l_old = objective(w_old, theta_left, theta_right, X_train, y)
        l_new = objective(w_new, theta_left, theta_right, X_train, y)
        l.append(objective(w_new, theta_left, theta_right, X_train, y))
        if it >= max_iter:
            break
    return theta_left, theta_right, w_new


class Node(HHCARTNode):
    def get_child(self, datum):
        if self.is_leaf:
            raise Exception("Leaf node does not have children.")
        X = deepcopy(datum)

        if X.dot(np.array(self._weights).T) >= 0:
            return self.left_child
        else:
            return self.right_child


class ContinuouslyOptimizedObliqueRegressionTreeFast(BaseEstimator):
    def __init__(
        self,
        impurity,
        segmentor,
        nu=1.0,
        thau=10,
        max_iter=100,
        tol=1e-6,
        step=0.1,
        **kwargs,
    ):
        self.impurity = impurity
        self.segmentor = segmentor
        self.nu = nu
        self.thau = thau
        self.max_iter = max_iter
        self.tol = tol
        self.step = step
        self._max_depth = kwargs.get("max_depth", None)
        self._min_samples = kwargs.get("min_samples", 2)
        self._root = None
        self._nodes = []

    def fit(self, X, y):
        self._root = self._generate_node(X, y, 0)

    def _terminate(self, X, y, cur_depth):
        if self._max_depth != None and cur_depth == self._max_depth:
            # maximum depth reached.
            return True
        elif y.size < self._min_samples:
            # minimum number of samples reached.
            return True
        elif np.unique(y).size == 1:
            return True
        else:
            return False

    def _generate_leaf_node(self, cur_depth, y):
        node = Node(cur_depth, y, is_leaf=True)
        self._nodes.append(node)
        return node

    def _generate_node(self, X, y, cur_depth):
        if self._terminate(X, y, cur_depth):
            return self._generate_leaf_node(cur_depth, y)
        else:
            impurity, sr, left_indices, right_indices = self.segmentor(
                X, y, self.impurity
            )

            X_stack = np.hstack((X, np.ones(X.shape[0])[:, np.newaxis]))

            theta_left, theta_right, w_new = optimization(
                X_stack,
                y,
                left_indices,
                right_indices,
                sr,
                self.tol,
                self.thau,
                self.step,
                self.nu,
                self.max_iter,
            )

            weights = w_new
            assert len(weights) == (
                X.shape[1] + 1
            ), f"{len(weights), (X.shape[1] + 1)}"

            # generate indices
            mask = (weights.dot(X_stack.T) >= 0).flatten()
            left_indices = np.arange(0, len(X))[mask]
            right_indices = np.arange(0, len(X))[np.logical_not(mask)]

            if np.sum(mask) == 0 or np.sum(np.logical_not(mask)) == 0:
                return self._generate_leaf_node(cur_depth, y)

            X_left, y_left = X[left_indices], y[left_indices]
            X_right, y_right = X[right_indices], y[right_indices]

            node = Node(
                cur_depth,
                y,
                split_rules=sr,
                weights=weights,
                left_child=self._generate_node(X_left, y_left, cur_depth + 1),
                right_child=self._generate_node(
                    X_right, y_right, cur_depth + 1
                ),
                is_leaf=False,
            )
            self._nodes.append(node)
            return node

    def predict(self, X):
        X = np.hstack((X, np.ones(X.shape[0])[:, np.newaxis]))

        def predict_single(datum):
            cur_node = self._root
            while not cur_node.is_leaf:
                cur_node = cur_node.get_child(datum)
            return cur_node.label

        if not self._root:
            raise Exception("Decision tree has not been trained.")
        size = X.shape[0]
        predictions = np.empty((size,), dtype=float)
        for i in range(size):
            predictions[i] = predict_single(X[i, :])
        return predictions
