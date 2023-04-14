import time
import argparse
import os
from scipy.io import arff
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from fimdlp.mdlp import FImdlp
from fimdlp.cppfimdlp import CArffFiles

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
arff = CArffFiles()
arff.load(bytes(f"{file_name}.arff", "utf-8"))
X = arff.get_X()
y = arff.get_y()
attributes = arff.get_attributes()
attributes = [x[0].decode() for x in attributes]
df = pd.DataFrame(X, columns=attributes)
class_name = arff.get_class_name().decode()
df[class_name] = y
test = FImdlp(
    min_length=args.min_length,
    max_depth=args.max_depth,
    max_cuts=args.max_cuts,
)
now = time.time()
test.fit(X, y)
fit_time = time.time()
print(f"Fitting ....: {fit_time - now:7.5f} seconds")
now = time.time()
Xt = test.transform(X)
print(f"Transforming: {time.time() - now:7.5f} seconds")
cut_points = test.get_cut_points()
for i, cuts in enumerate(cut_points):
    print(f"Cut points for feature {attributes[i]}: {cuts}")
    print(f"Min: {min(X[:, i]):6.4f} Max: {max(X[:, i]):6.4f}")
num_cuts = sum([len(x) for x in cut_points])
print(f"Total cut points ...: {num_cuts}")
print(f"Total feature states: {num_cuts + len(attributes)}")
clf = RandomForestClassifier(random_state=0)
print(
    "Random Forest score with discretized data: ", clf.fit(Xt, y).score(Xt, y)
)
