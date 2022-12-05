import unittest
import sklearn
from sklearn.datasets import load_iris
import numpy as np
from ..mdlp import FImdlp


class FImdlpTest(unittest.TestCase):
    def test_init(self):
        clf = FImdlp()
        self.assertTrue(clf.proposal)
        clf = FImdlp(proposal=False)
        self.assertFalse(clf.proposal)

    def test_fit(self):
        clf = FImdlp()
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
            [4.900000095367432, 5.0, 5.099999904632568, 5.400000095367432],
            [2.6999998092651367, 2.9000000953674316],
            [2.3499999046325684, 4.5],
            [0.75, 1.399999976158142, 1.5],
        ]
        self.assertListEqual(expected, clf.get_cut_points())
        with self.assertRaises(ValueError):
            clf.fit([[1, 2], [3, 4]], [1, 2, 3])

    def test_transform(self):
        clf = FImdlp()
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
            [4, 2, 2, 2],
            [4, 0, 1, 1],
            [1, 0, 1, 1],
            [4, 1, 1, 1],
            [4, 2, 1, 1],
            [4, 1, 1, 1],
        ]
        self.assertTrue(np.array_equal(clf.transform(X[90:97]), expected))
        with self.assertRaises(ValueError):
            clf.transform([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(sklearn.exceptions.NotFittedError):
            clf = FImdlp()
            clf.transform([[1, 2], [3, 4]])
