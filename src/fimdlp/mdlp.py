import numpy as np
from .cppfimdlp import CFImdlp, factorize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from joblib import Parallel, delayed


class FImdlp(TransformerMixin, BaseEstimator):
    def __init__(self, algorithm=0, n_jobs=-1):
        self.algorithm = algorithm
        self.n_jobs = n_jobs

    """Fayyad - Irani MDLP discretization algorithm based implementation.

    Parameters
    ----------
    algorithm : int, default=0
        The type of algorithm to use computing the cut points.
        0 - Definitive implementation
        1 - Alternative proposal
        2 - Classic proposal
    n_jobs : int, default=-1
        The number of jobs to run in parallel. :meth:`fit` and
        :meth:`transform`, are parallelized over the features. ``-1`` means
        using all cores available.

    Attributes
    ----------
    n_features_in_ : int
        The number of features of the data passed to :meth:`fit`.
    discretizer_ : list
        The list of discretizers, one for each feature.
    cut_points_ : list
        The list of cut points for each feature.
    X_ : array, shape (n_samples, n_features)
        the samples used to fit
    y_ : array, shape(n_samples,)
        the labels used to fit
    features_ : list
        the list of features to be discretized
    """

    def _more_tags(self):
        return {"preserves_dtype": [np.int32], "requires_y": True}

    def _check_args(self, X, y, expected_args, kwargs):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Default values
        self.features_ = [i for i in range(X.shape[1])]
        for key, value in kwargs.items():
            if key in expected_args:
                setattr(self, f"{key}_", value)
            else:
                raise ValueError(f"Unexpected argument: {key}")
        if len(self.features_) > X.shape[1]:
            raise ValueError(
                "Number of features does not match the number of columns in X"
            )
        if type(self.features_) != list:
            raise ValueError("features must be a list")
        self.features_.sort()
        if list(set(self.features_)) != self.features_:
            raise ValueError("Features must be unique")
        if max(self.features_) >= X.shape[1]:
            raise ValueError("Feature index out of range")
        return X, y

    def _update_params(self, X, y):
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.n_classes_ = self.classes_.shape[0]
        self.n_features_in_ = X.shape[1]

    def fit(self, X, y, **kwargs):
        """A reference implementation of a fitting function for a transformer.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The training input samples.
        y : array, shape (n_samples,)
            the labels used to fit
        features : list, default=[i for i in range(n_features)]
            The list of features to be discretized.
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = self._check_args(
            X, y, expected_args=["features"], kwargs=kwargs
        )
        self._update_params(X, y)
        self.X_ = X
        self.y_ = y
        self.discretizer_ = [None] * self.n_features_in_
        self.cut_points_ = [None] * self.n_features_in_
        Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(self._fit_discretizer)(feature)
            for feature in range(self.n_features_in_)
        )
        return self

    def _fit_discretizer(self, feature):
        if feature in self.features_:
            self.discretizer_[feature] = CFImdlp(algorithm=self.algorithm)
            self.discretizer_[feature].fit(self.X_[:, feature], self.y_)
            self.cut_points_[feature] = self.discretizer_[
                feature
            ].get_cut_points()
        else:
            self.discretizer_[feature] = None
            self.cut_points_[feature] = []

    def _discretize_feature(self, feature, X, result):
        if feature in self.features_:
            result[:, feature] = np.searchsorted(self.cut_points_[feature], X)
        else:
            result[:, feature] = X

    def transform(self, X):
        """Discretize X values.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the discretized values of ``X``.
        """
        # Check is fit had been called
        check_is_fitted(self, "n_features_in_")
        # Input validation
        X = check_array(X)
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "Shape of input is different from what was seen in `fit`"
            )
        if len(self.features_) == self.n_features_in_:
            result = np.zeros_like(X, dtype=np.int32) - 1
        else:
            result = np.zeros_like(X) - 1
        Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(self._discretize_feature)(feature, X[:, feature], result)
            for feature in range(self.n_features_in_)
        )
        return result

    def factorize(self, yy):
        """Factorize the input labels

        Parameters
        ----------
        yy : array, shape (n_samples,)
            Labels to be factorized, MUST be bytes, i.e. b"0", b"1", ...

        Returns
        -------
        array, shape (n_samples,)
            Factorized labels
        """
        return factorize(yy)

    def get_cut_points(self):
        """Get the cut points for each feature.
        Returns
        -------
        result: list
            The list of cut points for each feature.
        """
        result = []
        for feature in range(self.n_features_in_):
            result.append(self.cut_points_[feature])
        return result


class MultiDiscretizer:
    def __init__(self, algorithm=0, n_jobs=-1):
        self.algorithm = algorithm
        self.n_jobs = n_jobs

    def initial_fit_transform(self, X, y):
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]
        self.discretizers_ = [None] * self.n_features_in_
        self.discretized_ = [None] * self.n_features_in_
        # self.yy_ = [None] * self.n_features_in_
        self.X_d_ = np.zeros_like(X, dtype=np.int32) - 1
        for feature in range(self.n_features_in_):
            self.discretizers_[feature] = FImdlp(
                algorithm=self.algorithm, n_jobs=self.n_jobs
            )
            self.discretized_[feature] = self.discretizers_[
                feature
            ].fit_transform(X[:, feature].reshape(-1, 1), y)
            # self.yy_[feature] = self.discretizers_[feature].factorize(y)
            self.X_d_[:, feature] = self.discretized_[feature].ravel()
        return self.X_d_

    def transform(self, X):
        X = check_array(X)
        if not hasattr(self, "discretizers_"):
            raise ValueError("Must call fit_transform first")
        result = np.zeros_like(X, dtype=np.int32) - 1
        for feature in range(self.n_features_in_):
            result[:, feature] = (
                self.discretizers_[feature]
                .transform(X[:, feature].reshape(-1, 1))
                .ravel()
            )
        return result

    def join_transform(self, features, target):
        """Join the selected features with the labels and discretize the values
        of the target variable
        join - fit - transform

        Parameters
        ----------
        features : [list]
            index of the features to join with the labels
        target : [int]
            index of the target variable to discretize
        """
        # Check is fit had been called
        check_is_fitted(self, "n_features_in_")
        if len(features) < 1 or len(features) > self.n_features_in_:
            raise ValueError(
                "Number of features must be in range [1, "
                f"{self.n_features_in_}]"
            )
        for feature in features:
            if feature < 0 or feature >= self.n_features_in_:
                raise ValueError(
                    f"Feature {feature} not in range [0, "
                    f"{self.n_features_in_})"
                )
        if target < 0 or target >= self.n_features_in_:
            raise ValueError(
                f"Target {target} not in range [0, {self.n_features_in_})"
            )
        y_join = [
            f"{str(item_y)}{''.join([str(x) for x in items_x])}".encode()
            for item_y, items_x in zip(self.y_, self.X_d_[:, features])
        ]
        self.yy_[target] = self.discretizer_.factorize(y_join)
        self.discretizers_[target] = FImdlp(
            algorithm=self.algorithm, n_jobs=self.n_jobs
        )
        self.discretized_[target] = self.discretizers_[target].fit_transform(
            self.X_[:, target].reshape(-1, 1), self.yy_[target]
        )
        return self.discretized_[target]


# from sklearn.datasets import load_wine
# X, y = load_wine(return_X_y=True)
# from fimdlp.mdlp import MultiDiscretizer
# clf = MultiDiscretizer()
# clf.fit(X, y)
# clf.join_transform([1, 3, 5], 7)
