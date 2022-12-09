from sklearn.datasets import load_iris
from fimdlp.mdlp import FImdlp
from fimdlp.cppfimdlp import CFImdlp
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time

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
