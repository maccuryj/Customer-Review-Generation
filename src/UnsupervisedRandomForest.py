import numpy as np
from numpy.random import choice
from sklearn.datasets import load_iris, make_classification
import sklearn.ensemble._forest as _forest
from sklearn.cluster import DBSCAN
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.utils.validation import check_is_fitted
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import threading
from joblib import Parallel, delayed
import matplotlib.pyplot as plt



def _accumulate_prox(apply, X, out, lock):
    """
    This is a utility function for joblib's Parallel.
    It can't go locally in UnsupervisedRandomForest, because joblib
    complains that it cannot pickle it when placed there.
    """
    leaves = apply(X, check_input=False)

    with lock:
        out += np.equal.outer(leaves, leaves)

class UnsupervisedRandomForest(_forest.ForestClassifier):
    def __init__(self,
                    n_estimators=100,
                    criterion="gini",
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    min_weight_fraction_leaf=0.,
                    max_features="auto",
                    max_leaf_nodes=None,
                    min_impurity_decrease=0.,
                    min_impurity_split=None,
                    bootstrap=True,
                    oob_score=False,
                    n_jobs=None,
                    random_state=None,
                    verbose=0,
                    warm_start=False,
                    class_weight=None,
                    ccp_alpha=0.0,
                    max_samples=None):
        super().__init__(
            base_estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                                "min_samples_leaf", "min_weight_fraction_leaf",
                                "max_features", "max_leaf_nodes",
                                "min_impurity_decrease", "min_impurity_split",
                                "random_state", "ccp_alpha"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            max_samples=max_samples)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.ccp_alpha = ccp_alpha

    def _sample_synthetic(self, X):
        """
        Create synthetic data of the same shape as X
        Synthetic data points are created by sampling from
        univariate distributions of features across X
        """
        n_samples = X.shape[0]
        self.y = np.concatenate((np.ones(n_samples), np.zeros(n_samples)))
        
        random_state = _forest.check_random_state(self.random_state)        

        X_synth = np.asarray([np.apply_along_axis(random_state.choice, 0, X) for _ in range(n_samples)])
        self.X = np.concatenate((X, X_synth))

        return self.X, self.y

    def fit(self, X):
        """
        Forest of trees is created and fit according to
        concatenated original and synthetic data

        Args:
            X (np.array(2D)): Original data without labels
        """
        self.n_classes_ = 2
        self.n_features_ = X.shape[1]

        X, y = self._sample_synthetic(X)
        return _forest.BaseForest.fit(self, X, y)

    def create_proximity(self, X):
        """
        Create proximity matrix by processing data points with every single estimator
        and computing how often data point pairs land in the same leaf

        """
        _forest.check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _forest._partition_estimators(self.n_estimators, self.n_jobs)
        # avoid storing the output of every estimator by summing them here
        self.proximity_matrix = np.zeros((X.shape[0], X.shape[0]), dtype=np.int16).reshape(X.shape[0], X.shape[0])
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose,
                 **_joblib_parallel_args(require="sharedmem"))(
                delayed(_accumulate_prox)(e.apply, X, self.proximity_matrix,
                                            lock)
            for e in self.estimators_)

        #Normalize proximity matrix
        self.proximity_matrix = np.divide(self.proximity_matrix, self.n_estimators, dtype=np.half)
        self.proximity_matrix = 1-self.proximity_matrix

        return self.proximity_matrix

    def proximity_clustering(self, X):
        clustering = DBSCAN(eps=0.6, min_samples = 4, metric='precomputed').fit(self.proximity_matrix)
        core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
        core_samples_mask[clustering.core_sample_indices_] = True
        labels = clustering.labels_

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            xy = X[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=14)

            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=6)

        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()

        #def



