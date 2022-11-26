import numpy as np
from sklearn.datasets import load_iris
from fimdlp import CFImdlp

data = load_iris()
X = data.data
y = data.target
features = data.feature_names
test = CFImdlp()
print("Cut points for each feature in Iris dataset:")
for i in range(0, X.shape[1]):
    data = np.sort(X[:, i])
    Xcutpoints = test.cut_points(data, y)
    print(f"{features[i]:20s}: {Xcutpoints}")
