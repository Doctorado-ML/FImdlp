from sklearn.datasets import load_iris
from fimdlp.mdlp import FImdlp
from fimdlp.cppfimdlp import CFImdlp

data = load_iris()
X = data.data
y = data.target
features = data.feature_names
test = FImdlp()
test.fit(X, y, features=features).transform(X)
