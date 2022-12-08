import numpy as np
from math import log2
from types import SimpleNamespace


class PyFImdlp:
    def __init__(self, proposal=True, debug=False):
        self.proposal = proposal
        self.n_features_ = None
        self.X_ = None
        self.y_ = None
        self.debug = debug
        self.features_ = None
        self.cut_points_ = []
        self.entropy_cache = {}
        self.information_gain_cache = {}

    def fit(self, X, y):
        self.n_features_ = len(X)
        self.indices_ = np.argsort(X)
        self.use_indices = False
        X = [
            4.3,
            4.4,
            4.4,
            4.4,
            4.5,
            4.6,
            4.6,
            4.6,
            4.6,
            4.7,
            4.7,
            4.8,
            4.8,
            4.8,
            4.8,
            4.8,
            4.9,
            4.9,
            4.9,
            4.9,
            4.9,
            4.9,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5.1,
            5.1,
            5.1,
            5.1,
            5.1,
            5.1,
            5.1,
            5.1,
            5.1,
            5.2,
            5.2,
            5.2,
            5.2,
            5.3,
            5.4,
            5.4,
            5.4,
            5.4,
            5.4,
            5.4,
            5.5,
            5.5,
            5.5,
            5.5,
            5.5,
            5.5,
            5.5,
            5.6,
            5.6,
            5.6,
            5.6,
            5.6,
            5.6,
            5.7,
            5.7,
            5.7,
            5.7,
            5.7,
            5.7,
            5.7,
            5.7,
            5.8,
            5.8,
            5.8,
            5.8,
            5.8,
            5.8,
            5.8,
            5.9,
            5.9,
            5.9,
            6,
            6,
            6,
            6,
            6,
            6,
            6.1,
            6.1,
            6.1,
            6.1,
            6.1,
            6.1,
            6.2,
            6.2,
            6.2,
            6.2,
            6.3,
            6.3,
            6.3,
            6.3,
            6.3,
            6.3,
            6.3,
            6.3,
            6.3,
            6.4,
            6.4,
            6.4,
            6.4,
            6.4,
            6.4,
            6.4,
            6.5,
            6.5,
            6.5,
            6.5,
            6.5,
            6.6,
            6.6,
            6.7,
            6.7,
            6.7,
            6.7,
            6.7,
            6.7,
            6.7,
            6.7,
            6.8,
            6.8,
            6.8,
            6.9,
            6.9,
            6.9,
            6.9,
            7,
            7.1,
            7.2,
            7.2,
            7.2,
            7.3,
            7.4,
            7.6,
            7.7,
            7.7,
            7.7,
            7.7,
            7.9,
        ]
        y = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            2,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            2,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            2,
            0,
            1,
            1,
            2,
            0,
            1,
            2,
            1,
            2,
            2,
            1,
            1,
            2,
            1,
            1,
            1,
            2,
            1,
            2,
            2,
            1,
            1,
            1,
            1,
            2,
            2,
            1,
            1,
            2,
            2,
            1,
            2,
            2,
            1,
            2,
            1,
            2,
            2,
            1,
            2,
            2,
            2,
            1,
            2,
            2,
            2,
            1,
            2,
            2,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            1,
            1,
            1,
            2,
            2,
            1,
            2,
            1,
            2,
            2,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
        ]
        # self.X_ = X[self.indices_] if not self.use_indices else X
        # self.y_ = y[self.indices_] if not self.use_indices else y
        self.X_ = X
        self.y_ = y
        self.compute_cut_points(0, len(y))
        return self

    def get_cut_points(self):
        return sorted(list(set([cut.value for cut in self.cut_points_])))

    def compute_cut_points(self, start, end):
        # print((start, end))
        cut = self.get_candidate(start, end)
        if cut.value is None:
            return
        print("cut: ", cut.value, " index: ", cut.index)
        if self.mdlp(cut, start, end):
            print("¡Ding!", cut.value, cut.index)
            self.cut_points_.append(cut)
        self.compute_cut_points(start, cut.index)
        self.compute_cut_points(cut.index, end)

    def mdlp(self, cut, start, end):
        N = end - start
        k = self.num_classes(start, end)
        k1 = self.num_classes(start, cut.index)
        k2 = self.num_classes(cut.index, end)
        ent = self.entropy(start, end)
        ent1 = self.entropy(start, cut.index)
        ent2 = self.entropy(cut.index, end)
        ig = self.information_gain(start, cut.index, end)
        delta = log2(pow(3, k) - 2, 2) - (
            float(k) * ent - float(k1) * ent1 - float(k2) * ent2
        )
        term = 1 / N * (log2(N - 1, 2) + delta)
        print("start: ", start, " cut: ", cut.index, " end: ", end)
        print(
            "k=",
            k,
            " k1=",
            k1,
            " k2=",
            k2,
            " ent=",
            ent,
            " ent1=",
            ent1,
            " ent2=",
            ent2,
        )
        print("ig=", ig, " delta=", delta, " N ", N, " term ", term)
        return ig > term

    def num_classes(self, start, end):
        n_classes = set()
        for i in range(start, end):
            n_classes.add(
                self.y_[self.indices_[i]] if self.use_indices else self.y_[i]
            )
        return len(n_classes)

    def get_candidate(self, start, end):
        """Return the best cutpoint candidate for the given range.

        Parameters
        ----------
        start : int
            Start of the range.
        end : int
            End of the range.

        Returns
        -------
        candidate : SimpleNamespace with attributes index and value
            value == None if no candidate is found.
        """
        candidate = SimpleNamespace()
        candidate.value = None
        minEntropy = float("inf")
        for idx in range(start + 1, end):
            condition = (
                self.y_[self.indices_[idx]] == self.y_[self.indices_[idx - 1]]
                if self.use_indices
                else self.y_[idx] == self.y_[idx - 1]
            )
            if condition:
                continue
            entropy_left = self.entropy(start, idx)
            entropy_right = self.entropy(idx, end)
            entropy_cut = entropy_left + entropy_right
            print(
                "idx: ",
                idx,
                " entropy_left: ",
                entropy_left,
                " entropy_right : ",
                entropy_right,
                " -> ",
                start,
                " ",
                end,
            )
            if entropy_cut < minEntropy:
                minEntropy = entropy_cut
                candidate.index = idx
                if self.use_indices:
                    candidate.value = (
                        self.X_[self.indices_[idx]]
                        + self.X_[self.indices_[idx - 1]]
                    ) / 2
                else:
                    candidate.value = (self.X_[idx] + self.X_[idx - 1]) / 2
        return candidate

    def entropy(self, start, end) -> float:
        n_labels = end - start
        if n_labels <= 1:
            return 0
        if (start, end) in self.entropy_cache:
            return self.entropy_cache[(start, end)]
        if self.use_indices:
            counts = np.bincount(self.y_[self.indices_[start:end]])
        else:
            counts = np.bincount(self.y_[start:end])
        proportions = counts / n_labels
        n_classes = np.count_nonzero(proportions)
        if n_classes <= 1:
            return 0
        entropy = 0.0
        # Compute standard entropy.
        for prop in proportions:
            if prop != 0.0:
                entropy -= prop * log2(prop, 2)
        self.entropy_cache[(start, end)] = entropy
        return entropy

    def information_gain(self, start, cut, end):
        if (start, cut, end) in self.information_gain_cache:
            return self.information_gain_cache[(start, cut, end)]
        labels = end - start
        if labels == 0:
            return 0.0
        entropy = self.entropy(start, end)
        card_left = cut - start
        entropy_left = self.entropy(start, cut)
        card_right = end - cut
        entropy_right = self.entropy(cut, end)
        result = (
            entropy
            - (card_left / labels) * entropy_left
            - (card_right / labels) * entropy_right
        )
        self.information_gain_cache[(start, cut, end)] = result
        return result
