import numpy as np
from .cppfimdlp import CFImdlp, factorize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from joblib import Parallel, delayed
from ._version import __version__

# from ._version import __version__


class FImdlp(TransformerMixin, BaseEstimator):
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs

    """Fayyad - Irani MDLP discretization algorithm based implementation.

    Parameters
    ----------
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

    @staticmethod
    def get_version():
        return f"{__version__}({CFImdlp().get_version().decode()})"

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
            self.discretizer_[feature] = CFImdlp()
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

    @staticmethod
    def factorize(yy):
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

    def get_states_feature(self, feature):
        """Return the states a feature can take

        Parameters
        ----------
        feature : int
            feature to get the states

        Returns
        -------
        list
            states of the feature
        """
        if feature in self.features_:
            return list(range(len(self.cut_points_[feature]) + 1))
        return None

    def join_fit(self, features, target, data):
        """Join the selected features with the labels and fit the discretizer
        of the target variable
        join - fit - transform

        Parameters
        ----------
        features : [list]
            index of the features to join with the labels
        target : [int]
            index of the target variable to discretize
        data: [array] shape (n_samples, n_features)
            dataset that contains the features to join

        Returns
        -------
        result: np.array
            The target variable newly discretized
        """
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
        if target in features:
            raise ValueError("Target cannot in features to join")
        y_join = [
            f"{str(item_y)}{''.join([str(x) for x in items_x])}".encode()
            for item_y, items_x in zip(self.y_, data[:, features])
        ]
        self.y_join_ = y_join
        self.discretizer_[target].fit(self.X_[:, target], factorize(y_join))
        self.cut_points_[target] = self.discretizer_[target].get_cut_points()
        # return the discretized target variable with the new cut points
        return np.searchsorted(self.cut_points_[target], self.X_[:, target])
