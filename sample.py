from sklearn.datasets import load_iris
from fimdlp.mdlp import FImdlp
from fimdlp.cppfimdlp import CFImdlp

data = load_iris()
X = data.data
y = data.target
features = data.feature_names
test = FImdlp()
# Xcutpoints = test.fit(X, y, features=features).transform(X)
clf = CFImdlp(debug=True)
print("Cut points for feature 0 in Iris dataset:")
print(clf.cut_points(X[:, 0], y))
