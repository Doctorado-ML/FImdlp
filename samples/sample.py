import time
import argparse
import os
from scipy.io import arff
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from fimdlp.mdlp import FImdlp

datasets = {
    "mfeat-factors": True,
    "iris": True,
    "glass": True,
    "liver-disorders": True,
    "letter": True,
    "kdd_JapaneseVowels": False,
}

ap = argparse.ArgumentParser()
ap.add_argument(
    "--min_length", type=int, default=3, help="Minimum length of interval"
)
ap.add_argument("--max_depth", type=int, default=9999, help="Maximum depth")
ap.add_argument(
    "--max_cuts", type=float, default=0, help="Maximum number of cut points"
)
ap.add_argument("dataset", type=str, choices=datasets.keys())
args = ap.parse_args()
relative = "" if os.path.isdir("src") else ".."
file_name = os.path.join(
    relative, "src", "cppmdlp", "tests", "datasets", args.dataset
)
data = arff.loadarff(file_name + ".arff")
df = pd.DataFrame(data[0])
class_column = -1 if datasets[args.dataset] else 0
class_name = df.columns.to_list()[class_column]
X = df.drop(class_name, axis=1)
y, _ = pd.factorize(df[class_name])
X = X.to_numpy()
test = FImdlp(
    min_length=args.min_length,
    max_depth=args.max_depth,
    max_cuts=args.max_cuts,
)
now = time.time()
test.fit(X, y)
fit_time = time.time()
print("Fitting: ", fit_time - now)
now = time.time()
Xt = test.transform(X)
print("Transforming: ", time.time() - now)
print(test.get_cut_points())
clf = RandomForestClassifier(random_state=0)
print(
    "Random Forest score with discretized data: ", clf.fit(Xt, y).score(Xt, y)
)
