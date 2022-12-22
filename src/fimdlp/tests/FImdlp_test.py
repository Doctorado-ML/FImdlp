import unittest
import sklearn
from sklearn.datasets import load_iris
import numpy as np
from ..mdlp import FImdlp
from .. import version
from .._version import __version__


class FImdlpTest(unittest.TestCase):
    def test_version(self):
        self.assertEqual(version(), __version__)

    def test_init(self):
        clf = FImdlp()
        self.assertEqual(-1, clf.n_jobs)
        self.assertEqual(0, clf.algorithm)
        clf = FImdlp(algorithm=1, n_jobs=7)
        self.assertEqual(1, clf.algorithm)
        self.assertEqual(7, clf.n_jobs)

    def test_fit_definitive(self):
        clf = FImdlp(algorithm=0)
        clf.fit([[1, 2], [3, 4]], [1, 2])
        self.assertEqual(clf.n_features_, 2)
        self.assertListEqual(clf.X_.tolist(), [[1, 2], [3, 4]])
        self.assertListEqual(clf.y_.tolist(), [1, 2])
        self.assertListEqual([[2.0], [3.0]], clf.get_cut_points())
        X, y = load_iris(return_X_y=True)
        clf.fit(X, y)
        self.assertEqual(clf.n_features_, 4)
        self.assertTrue(np.array_equal(X, clf.X_))
        self.assertTrue(np.array_equal(y, clf.y_))
        expected = [
            [5.449999809265137, 6.25],
            [2.8499999046325684, 3.0, 3.049999952316284, 3.3499999046325684],
            [2.450000047683716, 4.75, 5.050000190734863],
            [0.800000011920929, 1.4500000476837158, 1.75],
        ]
        self.assertListEqual(expected, clf.get_cut_points())
        self.assertListEqual([0, 1, 2, 3], clf.features_)
        clf.fit(X, y, features=[0, 2, 3])
        self.assertListEqual([0, 2, 3], clf.features_)

    def test_fit_alternative(self):
        clf = FImdlp(algorithm=1)
        clf.fit([[1, 2], [3, 4]], [1, 2])
        self.assertEqual(clf.n_features_, 2)
        self.assertListEqual(clf.X_.tolist(), [[1, 2], [3, 4]])
        self.assertListEqual(clf.y_.tolist(), [1, 2])
        self.assertListEqual([[2], [3]], clf.get_cut_points())
        X, y = load_iris(return_X_y=True)
        clf.fit(X, y)
        self.assertEqual(clf.n_features_, 4)
        self.assertTrue(np.array_equal(X, clf.X_))
        self.assertTrue(np.array_equal(y, clf.y_))

        expected = [
            [5.449999809265137, 5.75],
            [2.8499999046325684, 3.3499999046325684],
            [2.450000047683716, 4.75],
            [0.800000011920929, 1.75],
        ]
        self.assertListEqual(expected, clf.get_cut_points())
        self.assertListEqual([0, 1, 2, 3], clf.features_)
        clf.fit(X, y, features=[0, 2, 3])
        self.assertListEqual([0, 2, 3], clf.features_)

    def test_fit_Errors(self):
        clf = FImdlp()
        with self.assertRaises(ValueError):
            clf.fit([[1, 2], [3, 4]], [1, 2, 3])
        with self.assertRaises(ValueError):
            clf.fit([[1, 2], [3, 4]], [1, 2], features=["a", "b", "c"])
        with self.assertRaises(ValueError):
            clf.fit([[1, 2], [3, 4]], [1, 2], unexpected="class_name")
        with self.assertRaises(ValueError):
            clf.fit([[1, 2], [3, 4]], [1, 2], features="01")
        with self.assertRaises(ValueError):
            clf.fit([[1, 2], [3, 4]], [1, 2], features=[0, 0])
        with self.assertRaises(ValueError):
            clf.fit([[1, 2], [3, 4]], [1, 2], features=[0, 2])

    def test_fit_features(self):
        clf = FImdlp()
        clf.fit([[1, -2], [3, 4]], [1, 2], features=[0])
        res = clf.transform([[1, -2], [3, 4]])
        self.assertListEqual(res.tolist(), [[0, -2], [1, 4]])
        X, y = load_iris(return_X_y=True)
        X_expected = X[:, [0, 2]].copy()
        clf.fit(X, y, features=[1, 3])
        X_computed = clf.transform(X)
        self.assertListEqual(
            X_expected[:, 0].tolist(), X_computed[:, 0].tolist()
        )
        self.assertListEqual(
            X_expected[:, 1].tolist(), X_computed[:, 2].tolist()
        )
        self.assertEqual(X_computed.dtype, np.float64)

    def test_transform_definitive(self):
        clf = FImdlp(algorithm=0)
        clf.fit([[1, 2], [3, 4]], [1, 2])
        self.assertEqual(
            clf.transform([[1, 2], [3, 4]]).tolist(), [[0, 0], [1, 1]]
        )
        X, y = load_iris(return_X_y=True)
        clf.fit(X, y)
        self.assertEqual(clf.n_features_, 4)
        self.assertTrue(np.array_equal(X, clf.X_))
        self.assertTrue(np.array_equal(y, clf.y_))
        X_transformed = clf.transform(X)
        self.assertListEqual(
            X_transformed.tolist(), clf.fit(X, y).transform(X).tolist()
        )
        self.assertEqual(X_transformed.dtype, np.int32)
        expected = [
            [1, 0, 1, 1],
            [1, 1, 1, 1],
            [1, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]
        self.assertTrue(np.array_equal(clf.transform(X[90:97]), expected))
        with self.assertRaises(ValueError):
            clf.transform([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(sklearn.exceptions.NotFittedError):
            clf = FImdlp(algorithm=0)
            clf.transform([[1, 2], [3, 4]])

    def test_transform_alternative(self):
        clf = FImdlp(algorithm=1)
        clf.fit([[1, 2], [3, 4]], [1, 2])
        self.assertEqual(
            clf.transform([[1, 2], [3, 4]]).tolist(), [[0, 0], [1, 1]]
        )
        X, y = load_iris(return_X_y=True)
        clf.fit(X, y)
        self.assertEqual(clf.n_features_, 4)
        self.assertTrue(np.array_equal(X, clf.X_))
        self.assertTrue(np.array_equal(y, clf.y_))
        self.assertListEqual(
            clf.transform(X).tolist(), clf.fit(X, y).transform(X).tolist()
        )
        expected = [
            [1, 0, 1, 1],
            [2, 1, 1, 1],
            [2, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]
        self.assertTrue(np.array_equal(clf.transform(X[90:97]), expected))
        with self.assertRaises(ValueError):
            clf.transform([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(sklearn.exceptions.NotFittedError):
            clf = FImdlp(algorithm=1)
            clf.transform([[1, 2], [3, 4]])
