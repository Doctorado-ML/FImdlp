import numpy as np
from .cppfimdlp import CFImdlp, factorize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,
)
from joblib import Parallel, delayed
from ._version import __version__


class FImdlp(TransformerMixin, BaseEstimator):
    def __init__(self, n_jobs=-1, min_length=3, max_depth=1e6, max_cuts=0):
        self.n_jobs = n_jobs
        self.min_length = min_length
        self.max_depth = max_depth
        self.max_cuts = max_cuts

    """Fayyad - Irani MDLP discretization algorithm based implementation.

    Parameters
    ----------
    n_jobs : int, default=-1
        The number of jobs to run in parallel. :meth:`fit` and
        :meth:`transform`, are parallelized over the features. ``-1`` means
        using all cores available.
    min_length: int, default=3
        The minimum length of an interval to be considered to be discretized.
    max_depth: int, default=1e6
        The maximum depth of the discretization process.
    max_cuts: float, default=0
        The maximum number of cut points to be computed for each feature.

    Attributes
    ----------
    n_features_in_ : int
        The number of features of the data passed to :meth:`fit`.
    discretizer_ : list
        The list of discretizers, one for each feature.
    X_ : array, shape (n_samples, n_features)
        the samples used to fit
    y_ : array, shape(n_samples,)
        the labels used to fit
    features_ : list
        the list of features to be discretized
    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.transformer_tags.preserves_dtype = ["int32"]
        return tags

    def _check_args(self, X, y, expected_args, kwargs):
        # validate_data sets n_features_in_ and runs the standard X/y checks
        X, y = validate_data(self, X, y)
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
        self.efective_min_length_ = (
            self.min_length
            if self.min_length > 1
            else int(self.min_length * X.shape[0])
        )
        self.discretizer_ = [None] * self.n_features_in_
        # Lazy cache: filled on first call to get_cut_points / get_states_feature
        self._cut_points_cache_ = [None] * self.n_features_in_
        Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(self._fit_discretizer)(feature)
            for feature in range(self.n_features_in_)
        )
        # target of every feature. Start with -1 => y (see join_fit)
        self.target_ = [-1] * self.n_features_in_
        return self

    def _fit_discretizer(self, feature):
        if feature in self.features_:
            self.discretizer_[feature] = CFImdlp(
                min_length=self.efective_min_length_,
                max_depth=self.max_depth,
                max_cuts=self.max_cuts,
            )
            self.discretizer_[feature].fit(self.X_[:, feature], self.y_)
        # cut points are pulled lazily; ensure cache slot is empty
        self._cut_points_cache_[feature] = None

    def _discretize_feature(self, feature, X, result):
        if feature in self.features_:
            result[:, feature] = self.discretizer_[feature].transform(X)
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
        # validate_data with reset=False enforces the same n_features_in_
        # as seen during fit and emits the canonical sklearn error message.
        X = validate_data(self, X, reset=False)
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

    def _load_cut_points(self, feature):
        """Lazily fetch cut points for a feature from the C++ object.

        The C++ layer stores cut points as ``[vmin, c1, ..., cn, vmax]``; the
        first and last entries are sentinels used by ``transform`` and are
        stripped here so the public API exposes only the intermediate cuts
        (backwards-compatible with the pre-2.x layout).
        """
        cached = self._cut_points_cache_[feature]
        if cached is not None:
            return cached
        disc = self.discretizer_[feature]
        if disc is None:
            self._cut_points_cache_[feature] = []
            return self._cut_points_cache_[feature]
        raw = disc.get_cut_points()
        self._cut_points_cache_[feature] = (
            list(raw[1:-1]) if len(raw) >= 2 else []
        )
        return self._cut_points_cache_[feature]

    def get_cut_points(self):
        """Get the cut points for each feature.
        Returns
        -------
        result: list
            The list of cut points for each feature.
        """
        check_is_fitted(self, "n_features_in_")
        return [
            self._load_cut_points(feature)
            for feature in range(self.n_features_in_)
        ]

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
            return list(range(len(self._load_cut_points(feature)) + 1))
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
            raise ValueError("Target cannot be in features to join")
        y_join = [
            f"{str(item_y)}{''.join([str(x) for x in items_x])}".encode()
            for item_y, items_x in zip(self.y_, data[:, features])
        ]
        # Store in target_ the features used with class to discretize target
        self.target_[target] = features + [-1]
        self.y_join_ = y_join
        self.discretizer_[target].fit(self.X_[:, target], factorize(y_join))
        # invalidate lazy cache for the re-fitted feature
        self._cut_points_cache_[target] = None
        return self.discretizer_[target].transform(self.X_[:, target])

    def get_depths(self):
        res = [0] * self.n_features_in_
        for feature in self.features_:
            res[feature] = self.discretizer_[feature].get_depth()
        return res
