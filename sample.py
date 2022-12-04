from sklearn.datasets import load_iris
from fimdlp.mdlp import FImdlp
from fimdlp.cppfimdlp import CFImdlp
import numpy as np
from math import log


def entropy(y: np.array) -> float:
    """Compute entropy of a labels set

    Parameters
    ----------
    y : np.array
        set of labels

    Returns
    -------
    float
        entropy
    """
    n_labels = len(y)
    if n_labels <= 1:
        return 0
    counts = np.bincount(y)
    proportions = counts / n_labels
    n_classes = np.count_nonzero(proportions)
    if n_classes <= 1:
        return 0
    entropy = 0.0
    # Compute standard entropy.
    for prop in proportions:
        if prop != 0.0:
            entropy -= prop * log(prop, 2)
    return entropy


def information_gain(
    labels: np.array, labels_up: np.array, labels_dn: np.array
) -> float:
    imp_prev = entropy(labels)
    card_up = card_dn = imp_up = imp_dn = 0
    if labels_up is not None:
        card_up = labels_up.shape[0]
        imp_up = entropy(labels_up)
    if labels_dn is not None:
        card_dn = labels_dn.shape[0] if labels_dn is not None else 0
        imp_dn = entropy(labels_dn)
    samples = card_up + card_dn
    if samples == 0:
        return 0.0
    else:
        result = (
            imp_prev
            - (card_up / samples) * imp_up
            - (card_dn / samples) * imp_dn
        )
        return result


data = load_iris()
X = data.data
y = data.target
features = data.feature_names
# test = FImdlp()
# test.fit(X, y, features=features)
# test.transform(X)
# test.get_cut_points()
for proposal in [True, False]:
    X = data.data
    y = data.target
    print("*** Proposal: ", proposal)
    test = CFImdlp(debug=True, proposal=proposal)
    test.fit(X[:, 0], y)
    result = test.get_cut_points()
    for item in result:
        print(
            f"Class={item['classNumber']} - ({item['start']:3d}, "
            f"{item['end']:3d}) -> ({item['fromValue']:3.1f}, "
            f"{item['toValue']:3.1f}]"
        )
    print(test.get_discretized_values())
    print("+" * 40)
    X = np.array(
        [
            [5.1, 3.5, 1.4, 0.2],
            [5.2, 3.0, 1.4, 0.2],
            [5.3, 3.2, 1.3, 0.2],
            [5.4, 3.1, 1.5, 0.2],
        ]
    )
    y = np.array([0, 0, 0, 1])
    print(test.fit(X[:, 0], y).transform(X[:, 0]))
    result = test.get_cut_points()
    for item in result:
        print(
            f"Class={item['classNumber']} - ({item['start']:3d}, {item['end']:3d})"
            f" -> ({item['fromValue']:3.1f}, {item['toValue']:3.1f}]"
        )
    print("*" * 40)
# print(Xs, ys)
# print("**********************")
# test = [(0, 3), (4, 4), (5, 5), (6, 8), (9, 9)]
# print(ys)
# for start, end in test:
#     print("Testing ", start, end, ys[:end], ys[end:])
#     print("Information gain: ", information_gain(ys, ys[:end], ys[end:]))
# print(test.transform(X))
# print(X)
# print(indices)
# print(np.array(X)[indices])


# # k = test.cut_points(X[:, 0], y)
# # print(k)
# # k = test.cut_points_ant(X[:, 0], y)
# # print(k)
# # test.debug_points(X[:, 0], y)
# X = [5.7, 5.3, 5.2, 5.1, 5.0, 5.6, 5.1, 6.0, 5.1, 5.9]
# y = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
# indices = [4, 3, 6, 8, 2, 1, 5, 0, 9, 7]
# clf = CFImdlp(debug=True, proposal=False)
# clf.fit(X, y)
# print(clf.get_cut_points())
# y = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
# # To check
# indices2 = np.argsort(X)
# Xs = np.array(X)[indices2]
# ys = np.array(y)[indices2]
