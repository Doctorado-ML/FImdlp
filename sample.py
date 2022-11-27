from sklearn.datasets import load_iris
from fimdlp.mdlp import FImdlp

data = load_iris()
X = data.data
y = data.target
features = data.feature_names
test = FImdlp()
Xcutpoints = test.fit(X, y, features=features).transform(X)
