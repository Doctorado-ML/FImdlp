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
        self.assertFalse(clf.proposal)
        clf = FImdlp(proposal=True, n_jobs=7)
        self.assertTrue(clf.proposal)
        self.assertEqual(7, clf.n_jobs)

    def test_fit_proposal(self):
        clf = FImdlp(proposal=True)
        clf.fit([[1, 2], [3, 4]], [1, 2])
        self.assertEqual(clf.n_features_, 2)
        self.assertListEqual(clf.X_.tolist(), [[1, 2], [3, 4]])
        self.assertListEqual(clf.y_.tolist(), [1, 2])
        self.assertListEqual([[], []], clf.get_cut_points())
        X, y = load_iris(return_X_y=True)
        clf.fit(X, y)
        self.assertEqual(clf.n_features_, 4)
        self.assertTrue(np.array_equal(X, clf.X_))
        self.assertTrue(np.array_equal(y, clf.y_))
        expected = [
            [
                4.900000095367432,
                5.0,
                5.099999904632568,
                5.400000095367432,
                5.699999809265137,
            ],
            [2.6999998092651367, 2.9000000953674316, 3.1999998092651367],
            [2.3499999046325684, 4.5, 4.800000190734863],
            [0.75, 1.399999976158142, 1.5, 1.7000000476837158],
        ]
        self.assertListEqual(expected, clf.get_cut_points())
        self.assertListEqual([0, 1, 2, 3], clf.features_)
        clf.fit(X, y, features=[0, 2, 3])
        self.assertListEqual([0, 2, 3], clf.features_)

    def test_fit_original(self):
        clf = FImdlp(proposal=False)
        clf.fit([[1, 2], [3, 4]], [1, 2])
        self.assertEqual(clf.n_features_, 2)
        self.assertListEqual(clf.X_.tolist(), [[1, 2], [3, 4]])
        self.assertListEqual(clf.y_.tolist(), [1, 2])
        self.assertListEqual([[], []], clf.get_cut_points())
        X, y = load_iris(return_X_y=True)
        clf.fit(X, y)
        self.assertEqual(clf.n_features_, 4)
        self.assertTrue(np.array_equal(X, clf.X_))
        self.assertTrue(np.array_equal(y, clf.y_))

        expected = [
            [5.5, 5.800000190734863],
            [2.9000000953674316, 3.3499999046325684],
            [2.450000047683716, 4.800000190734863],
            [0.800000011920929, 1.7999999523162842],
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
        clf.fit([[1, 2], [3, 4]], [1, 2], features=[0])
        res = clf.transform([[1, 2], [3, 4]])
        self.assertListEqual(res.tolist(), [[0, 2], [0, 4]])

    def test_transform_original(self):
        clf = FImdlp(proposal=False)
        clf.fit([[1, 2], [3, 4]], [1, 2])
        self.assertEqual(
            clf.transform([[1, 2], [3, 4]]).tolist(), [[0, 0], [0, 0]]
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
            [0, 0, 1, 1],
            [2, 1, 1, 1],
            [1, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 1, 1],
            [1, 0, 1, 1],
        ]
        self.assertTrue(np.array_equal(clf.transform(X[90:97]), expected))
        with self.assertRaises(ValueError):
            clf.transform([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(sklearn.exceptions.NotFittedError):
            clf = FImdlp(proposal=False)
            clf.transform([[1, 2], [3, 4]])

    def test_transform_proposal(self):
        clf = FImdlp(proposal=True)
        clf.fit([[1, 2], [3, 4]], [1, 2])
        self.assertEqual(
            clf.transform([[1, 2], [3, 4]]).tolist(), [[0, 0], [0, 0]]
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
            [4, 0, 1, 1],
            [5, 2, 2, 2],
            [5, 0, 1, 1],
            [1, 0, 1, 1],
            [4, 1, 1, 1],
            [5, 2, 1, 1],
            [5, 1, 1, 1],
        ]
        self.assertTrue(np.array_equal(clf.transform(X[90:97]), expected))
        with self.assertRaises(ValueError):
            clf.transform([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(sklearn.exceptions.NotFittedError):
            clf = FImdlp(proposal=True)
            clf.transform([[1, 2], [3, 4]])
