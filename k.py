from sklearn.datasets import load_wine
from fimdlp.mdlp import FImdlp

X, y = load_wine(return_X_y=True)
trans = FImdlp()
Xt = trans.join_transform(X, y, 12)
print("X shape = ", X.shape)
print("Xt.shape=", Xt.shape)
print("Xt ", Xt[:10])
print("trans.X_ shape = ", trans.X_.shape)
print("trans.y_ ", trans.y_[:10])
print("y_join ", trans.y_join_[:10])
