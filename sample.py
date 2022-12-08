from sklearn.datasets import load_iris
from fimdlp.mdlp import FImdlp
from fimdlp.cppfimdlp import CFImdlp
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time
from math import log2

from scipy.io import arff
import pandas as pd

# class_name = "speaker"
# file_name = "kdd_JapaneseVowels.arff"
class_name = "class"
# file_name = "mfeat-factors.arff"
file_name = "letter.arff"
data = arff.loadarff(file_name)
df = pd.DataFrame(data[0])
df.dropna(axis=0, how="any", inplace=True)
dataset = df
X = df.drop(class_name, axis=1)
features = X.columns
class_name = class_name
y, _ = pd.factorize(df[class_name])
X = X.to_numpy()

# data = load_iris()
# X = data.data
# y = data.target
# features = data.feature_names


test = FImdlp()
now = time.time()
# test.fit(X, y, features=[i for i in (range(3, 14))])
test.fit(X, y)
fit_time = time.time()
print("Fitting: ", fit_time - now)
now = time.time()
Xt = test.transform(X)
print("Transforming: ", time.time() - now)
print(test.get_cut_points())

clf = RandomForestClassifier(random_state=0)
print(clf.fit(Xt, y).score(Xt, y))
print(Xt)
# for proposal in [True, False]:
#     X = data.data
#     y = data.target
#     print("*** Proposal: ", proposal)
#     test = CFImdlp(debug=True, proposal=proposal)
#     test.fit(X[:, 0], y)
#     result = test.get_cut_points()
#     for item in result:
#         print(
#             f"Class={item['classNumber']} - ({item['start']:3d}, "
#             f"{item['end']:3d}) -> ({item['fromValue']:3.1f}, "
#             f"{item['toValue']:3.1f}]"
#         )
#     print(test.get_discretized_values())
#     print("+" * 40)
#     X = np.array(
#         [
#             [5.1, 3.5, 1.4, 0.2],
#             [5.2, 3.0, 1.4, 0.2],
#             [5.3, 3.2, 1.3, 0.2],
#             [5.4, 3.1, 1.5, 0.2],
#         ]
#     )
#     y = np.array([0, 0, 0, 1])
#     print(test.fit(X[:, 0], y).transform(X[:, 0]))
#     result = test.get_cut_points()
#     for item in result:
#         print(
#             f"Class={item['classNumber']} - ({item['start']:3d}, {item['end']:3d})"
#             f" -> ({item['fromValue']:3.1f}, {item['toValue']:3.1f}]"
#         )
#     print("*" * 40)
# # print(Xs, ys)
# # print("**********************")
# # test = [(0, 3), (4, 4), (5, 5), (6, 8), (9, 9)]
# # print(ys)
# # for start, end in test:
# #     print("Testing ", start, end, ys[:end], ys[end:])
# #     print("Information gain: ", information_gain(ys, ys[:end], ys[end:]))
# # print(test.transform(X))
# # print(X)
# # print(indices)
# # print(np.array(X)[indices])


# # # k = test.cut_points(X[:, 0], y)
# # # print(k)
# # # k = test.cut_points_ant(X[:, 0], y)
# # # print(k)
# # # test.debug_points(X[:, 0], y)
# # X = [5.7, 5.3, 5.2, 5.1, 5.0, 5.6, 5.1, 6.0, 5.1, 5.9]
# # y = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
# # indices = [4, 3, 6, 8, 2, 1, 5, 0, 9, 7]
# # clf = CFImdlp(debug=True, proposal=False)
# # clf.fit(X, y)
# # print(clf.get_cut_points())
# # y = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
# # # To check
# # indices2 = np.argsort(X)
# # Xs = np.array(X)[indices2]
# # ys = np.array(y)[indices2]
# kdd_JapaneseVowels
