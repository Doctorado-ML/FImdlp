import numpy as np
from .cppfimdlp import CFImdlp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class FImdlp(TransformerMixin, BaseEstimator):
    """Fayyad - Irani MDLP discretization algorithm.

    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
    """

    def _check_params_fit(self, X, y, expected_args, kwargs):
        """Check the common parameters passed to fit"""
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.n_classes_ = self.classes_.shape[0]
        # Default values
        self.class_name_ = "class"
        self.features_ = [f"feature_{i}" for i in range(X.shape[1])]
        for key, value in kwargs.items():
            if key in expected_args:
                setattr(self, f"{key}_", value)
            else:
                raise ValueError(f"Unexpected argument: {key}")
        if len(self.features_) != X.shape[1]:
            raise ValueError(
                "Number of features does not match the number of columns in X"
            )
        return X, y

    def fit(self, X, y, **kwargs):
        """A reference implementation of a fitting function for a transformer.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = self._check_params_fit(
            X, y, expected_args=["class_name", "features"], kwargs=kwargs
        )

        self.n_features_ = X.shape[1]
        self.X_ = X
        self.y_ = y
        self.discretizer_ = CFImdlp(debug=False)
        return self

    def transform(self, X):
        """Discretize X values.
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the discretized values of ``X``.
        """
        # Check is fit had been called
        check_is_fitted(self, "n_features_")

        # Input validation
        X = check_array(X)
        if (X != self.X_).any():
            raise ValueError(
                "X values are not the same as the ones used to fit the model."
            )

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_:
            raise ValueError(
                "Shape of input is different from what was seen in `fit`"
            )
        print("Calculating cut points in python for first feature")
        yz = self.y_.copy()
        xz = X[:, 0].copy()
        xz = xz[np.argsort(X[:, 0])]
        yz = yz[np.argsort(X[:, 0])]
        cuts = []
        for i in range(1, len(yz)):
            if yz[i] != yz[i - 1] and xz[i - 1] < xz[i]:
                print(f"Cut point: ({xz[i-1]}, {xz[i]}) ({yz[i-1]}, {yz[i]})")
                cuts.append((xz[i] + xz[i - 1]) / 2)
        print("Cuts calculados en python: ", cuts)
        print("Cuts calculados en C++")
        print("Cut points for each feature in Iris dataset:")
        for i in range(0, self.n_features_):
            # datax = self.X_[np.argsort(self.X_[:, i]), i]
            # y_ = self.y_[np.argsort(self.X_[:, i])]
            datax = self.X_[:, i]
            y_ = self.y_
            Xcutpoints = self.discretizer_.cut_points(datax, y_)
            print(
                f"New ({len(Xcutpoints)}):{self.features_[i]:20s}: "
                f"{Xcutpoints}"
            )
            Xcutpoints = self.discretizer_.cut_points_ant(datax, y_)
            print(
                f"Ant ({len(Xcutpoints)}):{self.features_[i]:20s}: "
                f"{Xcutpoints}"
            )
        return X
