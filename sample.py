from sklearn.datasets import load_iris
from fimdlp.mdlp import FImdlp
from fimdlp.cppfimdlp import CFImdlp
import numpy as np

data = load_iris()
X = data.data
y = data.target
features = data.feature_names
test = FImdlp()
test.fit(X, y, features=features)
# test.transform(X)

test = CFImdlp(debug=False)
# k = test.cut_points(X[:, 0], y)
# print(k)
# k = test.cut_points_ant(X[:, 0], y)
# print(k)
test.debug_points(X[:, 0], y)

# X = np.array(
#     [
#         [5.1, 3.5, 1.4, 0.2],
#         [5.2, 3.0, 1.4, 0.2],
#         [5.3, 3.2, 1.3, 0.2],
#         [5.3, 3.1, 1.5, 0.2],
#     ]
# )
# y = np.array([0, 0, 0, 1])
# test.fit(X, y).transform(X)
