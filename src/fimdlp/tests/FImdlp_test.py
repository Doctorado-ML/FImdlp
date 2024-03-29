import unittest
import sklearn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.utils.estimator_checks import check_estimator
from ..cppfimdlp import CFImdlp, factorize, CArffFiles
from ..mdlp import FImdlp
from .. import __version__


class FImdlpTest(unittest.TestCase):
    delta = 1e-6  # same tolerance as in C++ code

    def test_version(self):
        clf = FImdlp()
        self.assertEqual(
            clf.get_version(),
            f"{__version__}({CFImdlp().get_version().decode()})",
        )

    def test_minimum_mdlp_version(self):
        mdlp_version = tuple(
            int(c) for c in CFImdlp().get_version().decode().split(".")[0:3]
        )
        minimum_mdlp_version = (1, 1, 2)
        self.assertTrue(mdlp_version >= minimum_mdlp_version)

    def test_init(self):
        clf = FImdlp()
        self.assertEqual(-1, clf.n_jobs)
        self.assertEqual(3, clf.min_length)
        self.assertEqual(1e6, clf.max_depth)
        clf = FImdlp(n_jobs=7, min_length=24, max_depth=17)
        self.assertEqual(7, clf.n_jobs)
        self.assertEqual(24, clf.min_length)
        self.assertEqual(17, clf.max_depth)

    def test_fit_definitive(self):
        clf = FImdlp()
        X, y = load_iris(return_X_y=True)
        clf.fit(X, y)
        self.assertEqual(clf.n_features_in_, 4)
        self.assertTrue(np.array_equal(X, clf.X_))
        self.assertTrue(np.array_equal(y, clf.y_))
        expected = [
            [5.45, 5.75],
            [2.75, 2.85, 2.95, 3.05, 3.35],
            [2.45, 4.75, 5.05],
            [0.8, 1.75],
        ]
        computed = clf.get_cut_points()
        for item_computed, item_expected in zip(computed, expected):
            for x_, y_ in zip(item_computed, item_expected):
                self.assertAlmostEqual(x_, y_, delta=self.delta)
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
        clf = FImdlp(n_jobs=-1)
        # Two samples doesn't have enough information to split
        clf.fit([[1, -2], [3, 4]], [1, 2], features=[0])
        self.assertListEqual(clf.get_cut_points(), [[], []])
        clf.fit([[1, -2], [3, 4], [5, 6]], [1, 2, 2], features=[0])
        self.assertListEqual(clf.get_cut_points(), [[2], []])
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

    def test_transform(self):
        clf = FImdlp()
        clf.fit([[1, 2], [3, 4], [5, 6]], [1, 2, 2])
        self.assertEqual(
            clf.transform([[1, 2], [3, 4]]).tolist(), [[0, 0], [1, 1]]
        )
        X, y = load_iris(return_X_y=True)
        clf.fit(X, y)
        self.assertEqual(clf.n_features_in_, 4)
        self.assertTrue(np.array_equal(X, clf.X_))
        self.assertTrue(np.array_equal(y, clf.y_))
        X_transformed = clf.transform(X)
        self.assertListEqual(
            X_transformed.tolist(), clf.fit(X, y).transform(X).tolist()
        )
        self.assertEqual(X_transformed.dtype, np.int32)
        expected = [
            [1, 0, 1, 1],
            [2, 3, 1, 1],
            [2, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 0, 1, 1],
            [1, 3, 1, 1],
            [1, 2, 1, 1],
        ]
        self.assertTrue(np.array_equal(clf.transform(X[90:97]), expected))
        with self.assertRaises(ValueError):
            clf.transform([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(sklearn.exceptions.NotFittedError):
            clf = FImdlp()
            clf.transform([[1, 2], [3, 4]])

    def test_cppfactorize(self):
        source = [
            b"f0",
            b"f1",
            b"f2",
            b"f3",
            b"f4",
            b"f5",
            b"f6",
            b"f1",
            b"f1",
            b"f7",
            b"f8",
        ]
        expected = [0, 1, 2, 3, 4, 5, 6, 1, 1, 7, 8]
        computed = factorize(source)
        self.assertListEqual(expected, computed)

    def test_join_fit(self):
        y = np.array([b"f0", b"f0", b"f2", b"f3", b"f3", b"f4", b"f4"])
        x = np.array(
            [
                [0, 1, 2, 3, 4, 5],
                [0, 2, 2, 3, 4, 5],
                [1, 2, 3, 4, 5, 5],
                [2, 3, 4, 5, 6, 6],
                [3, 4, 5, 6, 7, 7],
                [1, 2, 2, 3, 5, 7],
                [1, 3, 4, 4, 4, 7],
            ]
        )
        expected = [0, 1, 1, 2, 2, 1, 2]
        clf = FImdlp()
        clf.fit(x, factorize(y))
        computed = clf.join_fit([0, 2, 3, 4], 1, x)
        self.assertListEqual(computed.tolist(), expected)
        expected_y = [
            b"00234",
            b"00234",
            b"11345",
            b"22456",
            b"23567",
            b"31235",
            b"31444",
        ]
        self.assertListEqual(expected_y, clf.y_join_)

    def test_join_fit_error(self):
        y = np.array([b"f0", b"f0", b"f2", b"f3", b"f4"])
        x = np.array(
            [
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6],
                [3, 4, 5, 6, 7],
            ]
        )
        clf = FImdlp()
        clf.fit(x, factorize(y))
        with self.assertRaises(ValueError) as exception:
            clf.join_fit([], 1, x)
        self.assertEqual(
            str(exception.exception),
            "Number of features must be in range [1, 5]",
        )
        with self.assertRaises(ValueError) as exception:
            FImdlp().join_fit([0, 4], 1, x)
        self.assertTrue(
            str(exception.exception).startswith(
                "This FImdlp instance is not fitted yet."
            )
        )
        with self.assertRaises(ValueError) as exception:
            clf.join_fit([0, 5], 1, x)
        self.assertEqual(
            str(exception.exception),
            "Feature 5 not in range [0, 5)",
        )
        with self.assertRaises(ValueError) as exception:
            clf.join_fit([0, 2], 5, x)
        self.assertEqual(
            str(exception.exception),
            "Target 5 not in range [0, 5)",
        )
        with self.assertRaises(ValueError) as exception:
            clf.join_fit([0, 2], 2, x)
        self.assertEqual(
            str(exception.exception),
            "Target cannot be in features to join",
        )

    def test_factorize(self):
        y = np.array([b"f0", b"f0", b"f2", b"f3", b"f4"])
        clf = FImdlp()
        computed = clf.factorize(y)
        self.assertListEqual([0, 0, 1, 2, 3], computed)
        y = [b"f4", b"f0", b"f0", b"f2", b"f3"]
        clf = FImdlp()
        computed = clf.factorize(y)
        self.assertListEqual([0, 1, 1, 2, 3], computed)

    def test_join_fit_info(self):
        clf = FImdlp()
        X, y = load_iris(return_X_y=True)
        clf.fit(X, y)
        clf.join_fit([0, 2], 1, X)
        clf.join_fit([0, 3], 2, X)
        clf.join_fit([1, 2], 3, X)
        expected = [-1, [0, 2, -1], [0, 3, -1], [1, 2, -1]]
        self.assertListEqual(expected, clf.target_)

    @staticmethod
    def test_sklearn_transformer():
        for check, test in check_estimator(FImdlp(), generate_only=True):
            test(check)

    def test_states_feature(self):
        clf = FImdlp()
        X, y = load_iris(return_X_y=True)
        clf.fit(X, y)
        expected = []
        for i in [3, 6, 4, 3]:
            expected.append(list(range(i)))
        for feature in range(X.shape[1]):
            self.assertListEqual(
                expected[feature], clf.get_states_feature(feature)
            )

    def test_states_no_feature(self):
        clf = FImdlp()
        X, y = load_iris(return_X_y=True)
        clf.fit(X, y)
        self.assertIsNone(clf.get_states_feature(4))

    def test_MaxDepth(self):
        clf = FImdlp(max_depth=1)
        X, y = load_iris(return_X_y=True)
        clf.fit(X, y)
        expected_cutpoints = [
            [5.45],
            [3.35],
            [2.45],
            [0.8],
        ]
        expected_depths = [1] * 4
        self.assertListEqual(expected_depths, clf.get_depths())
        for expected, computed in zip(
            expected_cutpoints, clf.get_cut_points()
        ):
            for e, c in zip(expected, computed):
                self.assertAlmostEqual(e, c, delta=self.delta)

    def test_MinLength(self):
        clf = FImdlp(min_length=75)
        X, y = load_iris(return_X_y=True)
        clf.fit(X, y)
        expected_cutpoints = [
            [5.45, 5.75],
            [2.85, 3.35],
            [2.45, 4.75],
            [0.8, 1.75],
        ]
        expected_depths = [3, 2, 2, 2]
        self.assertListEqual(expected_depths, clf.get_depths())
        for expected, computed in zip(
            expected_cutpoints, clf.get_cut_points()
        ):
            for e, c in zip(expected, computed):
                self.assertAlmostEqual(e, c, delta=self.delta)

    def test_MinLengthMaxDepth(self):
        clf = FImdlp(min_length=75, max_depth=2)
        X, y = load_iris(return_X_y=True)
        clf.fit(X, y)
        expected_cutpoints = [
            [5.45, 5.75],
            [2.85, 3.35],
            [2.45, 4.75],
            [0.8, 1.75],
        ]
        expected_depths = [2, 2, 2, 2]
        self.assertListEqual(expected_depths, clf.get_depths())
        for expected, computed in zip(
            expected_cutpoints, clf.get_cut_points()
        ):
            for e, c in zip(expected, computed):
                self.assertAlmostEqual(e, c, delta=self.delta)

    def test_max_cuts(self):
        clf = FImdlp(max_cuts=1)
        X, y = load_iris(return_X_y=True)
        clf.fit(X, y)
        expected_cutpoints = [
            [5.45],
            [2.85],
            [2.45],
            [0.8],
        ]
        expected_depths = [3, 5, 4, 3]
        self.assertListEqual(expected_depths, clf.get_depths())
        for expected, computed in zip(
            expected_cutpoints, clf.get_cut_points()
        ):
            for e, c in zip(expected, computed):
                self.assertAlmostEqual(e, c, delta=self.delta)

    def test_ArffFiles(self):
        loader = CArffFiles()
        loader.load(b"src/cppmdlp/tests/datasets/iris.arff")
        X = loader.get_X()
        y = loader.get_y()
        expected = [
            (b"sepallength", b"REAL"),
            (b"sepalwidth", b"REAL"),
            (b"petallength", b"REAL"),
            (b"petalwidth", b"REAL"),
        ]
        self.assertListEqual(loader.get_attributes(), expected)
        self.assertListEqual(y[:10], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        expected = [
            b"5.1,3.5,1.4,0.2,Iris-setosa",
            b"4.9,3.0,1.4,0.2,Iris-setosa",
            b"4.7,3.2,1.3,0.2,Iris-setosa",
            b"4.6,3.1,1.5,0.2,Iris-setosa",
            b"5.0,3.6,1.4,0.2,Iris-setosa",
            b"5.4,3.9,1.7,0.4,Iris-setosa",
            b"4.6,3.4,1.4,0.3,Iris-setosa",
            b"5.0,3.4,1.5,0.2,Iris-setosa",
            b"4.4,2.9,1.4,0.2,Iris-setosa",
            b"4.9,3.1,1.5,0.1,Iris-setosa",
        ]
        self.assertListEqual(loader.get_lines()[:10], expected)
        expected_X = [
            [5.0999999, 3.5, 1.39999998, 0.2],
            [4.9000001, 3, 1.39999998, 0.2],
            [4.69999981, 3.20000005, 1.29999995, 0.2],
        ]
        for computed, expected in zip(X[:3].tolist(), expected_X):
            for c, e in zip(computed, expected):
                self.assertAlmostEqual(c, e, delta=self.delta)
