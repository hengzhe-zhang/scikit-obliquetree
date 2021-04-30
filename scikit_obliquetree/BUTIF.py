from copy import deepcopy

import numpy as np
from scipy.linalg import norm
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression


class Node:
    def __init__(self, objects, labels, **kwargs):

        self.objects = objects
        self.labels = labels
        self.centroid = objects.mean(axis=0)
        self.class_ = kwargs.get("class_", None)
        self.is_leaf = kwargs.get("is_leaf", False)
        self._weights = kwargs.get("weights", None)
        self._left_child = kwargs.get("left_child", None)
        self._right_child = kwargs.get("right_child", None)
        self._features_indices = kwargs.get("indices", None)

    def get_child(self, datum):
        if self.is_leaf:
            raise Exception("Leaf node does not have children.")
        X = deepcopy(datum)
        X = datum[self._features_indices]

        if X.dot(np.array(self._weights[:-1]).T) + self._weights[-1] < 0:
            return self.left_child
        else:
            return self.right_child

    @property
    def left_child(self):
        if self.is_leaf:
            raise Exception("Leaf node does not have split rule.")
        return self._left_child

    @property
    def right_child(self):
        if self.is_leaf:
            raise Exception("Leaf node does not have split rule.")
        return self._right_child


class Clustering(BaseEstimator):
    def __init__(self, method="k-means", n_clusters=6):
        self.method = method
        self.n_clusters = n_clusters

    def fit(self, X):
        if self.method == "k-means":
            clf = KMeans(n_clusters=self.n_clusters)
        else:
            raise Exception
        clf.fit(X)
        y = clf.predict(X)

        self.clusters_ = []
        for cluster in np.unique(y):
            self.clusters_.append(np.where(y == cluster)[0])

        self.clusters_ = np.array(self.clusters_, dtype=object)


class BUTIF(BaseEstimator):
    def __init__(
        self,
        clustering_method="k-means",
        max_leaf=5,
        linear_model=LogisticRegression(),
        task="classification",
        selector=None,
        best_k=None,
    ):
        self.clustering_method = clustering_method
        self.max_leaf = max_leaf
        self.clustering = Clustering(
            method=self.clustering_method, n_clusters=self.max_leaf
        )
        self.linear_model = linear_model
        self.selector = selector
        self._root = None
        self._nodes = []
        self._leaves = {}
        self.best_k = best_k
        self._k_features = best_k
        self.task = task

    def fit(self, X, y):
        def merging(L, classes_):
            if len(L) == 1:
                single_node = list(L.keys())[0]
                return L[single_node]
            else:
                dist_min = np.inf
                distance = 0
                for i in L.keys():
                    for j in L.keys():
                        if i == j:
                            continue
                        if L[i].class_ == L[j].class_:
                            continue

                        distance = norm(L[i].centroid - L[j].centroid)

                        if distance < dist_min:
                            dist_min = distance
                            i_node, j_node = i, j
                            left_child = L[i]
                            right_child = L[j]

                objects_ = np.vstack((left_child.objects, right_child.objects))
                labels_ = np.ones(objects_.shape[0])
                labels_[: left_child.objects.shape[0]] = -1.0
                labels_[left_child.objects.shape[0] :] = 1.0

                indices_ = np.arange(n_features)
                if self.selector is not None:
                    self.selector.fit(objects_, labels_)
                    indices_ = self.selector.scores_.argsort()[
                        : self._k_features
                    ]

                self.linear_model.fit(objects_[:, indices_], labels_)
                weights_ = np.zeros(len(indices_) + 1)
                weights_[:-1] = self.linear_model.coef_
                weights_[-1] = self.linear_model.intercept_

                labels_[: left_child.objects.shape[0]] = left_child.labels
                labels_[left_child.objects.shape[0] :] = right_child.labels

                new_node = Node(
                    objects_,
                    labels_,
                    left_child=left_child,
                    right_child=right_child,
                    class_=classes_ + 1,  # new meta class
                    indices=indices_,
                    weights=weights_,
                )
                del L[i_node]
                del L[j_node]
                L[i_node] = new_node
                return merging(L, classes_ + 1)

        n_objects, n_features = X.shape

        leaf_cnt = 0
        if self.task == "classification":
            self.classes_ = np.unique(y)
            for c in self.classes_:
                objects_in_c = X[y == c]

                self.clustering.fit(objects_in_c)
                partition = self.clustering.clusters_

                for cluster in partition:
                    leaf = Node(
                        objects_in_c[cluster],
                        c * np.ones(len(cluster)),
                        is_leaf=True,
                    )
                    leaf.class_ = c
                    self._leaves[str(leaf_cnt)] = leaf
                    leaf_cnt += 1
        else:
            # self.clustering.fit(X)
            self.clustering.fit(y.reshape(-1, 1))
            partition = self.clustering.clusters_
            for i, cluster in enumerate(partition):
                leaf = Node(
                    X[cluster.astype(int)],
                    i * np.ones(len(cluster)),
                    is_leaf=True,
                )
                leaf.class_ = np.mean(y[cluster.astype(int)])
                self._leaves[str(leaf_cnt)] = leaf
                leaf_cnt += 1
            self.classes_ = np.arange(len(partition))

        self._root = merging(self._leaves, len(self.classes_))

    def predict(self, X):
        def _predict_single(X):
            cur_node = self._root
            while not cur_node.is_leaf:
                cur_node = cur_node.get_child(X)
            return cur_node.class_

        if self._root is None:
            raise Exception("Decision tree has not been trained.")
        n_objects = X.shape[0]
        predictions = np.zeros((n_objects,), dtype=float)
        for i in range(n_objects):
            predictions[i] = _predict_single(X[i])
        return predictions
