#include <numeric>
#include <algorithm>
#include <set>
#include <cmath>
#include "CPPFImdlp.h"
#include "Metrics.h"

namespace mdlp {
    CPPFImdlp::CPPFImdlp(bool proposal):proposal(proposal), indices(indices_t()), X(samples_t()), y(labels_t()), metrics(Metrics(y, indices))
    {
    }
    CPPFImdlp::~CPPFImdlp()
        = default;

    CPPFImdlp& CPPFImdlp::fit(samples_t& X_, labels_t& y_)
    {
        X = X_;
        y = y_;
        cutPoints.clear();
        if (X.size() != y.size()) {
            throw invalid_argument("X and y must have the same size");
        }
        if (X.size() == 0 || y.size() == 0) {
            throw invalid_argument("X and y must have at least one element");
        }
        indices = sortIndices(X_);
        metrics.setData(y, indices);
        if (proposal)
            computeCutPointsProposal();
        else
            computeCutPoints(0, X.size());
        return *this;
    }
    void CPPFImdlp::computeCutPoints(size_t start, size_t end)
    {
        int cut;
        if (end - start < 2)
            return;
        cut = getCandidate(start, end);
        if (cut == -1 || !mdlp(start, cut, end)) {
            // cut.value == -1 means that there is no candidate in the interval
            // No boundary found, so we add both ends of the interval as cutpoints
            // because they were selected by the algorithm before
            if (start != 0)
                cutPoints.push_back((X[indices[start]] + X[indices[start - 1]]) / 2);
            if (end != X.size())
                cutPoints.push_back((X[indices[end]] + X[indices[end - 1]]) / 2);
            return;
        }
        computeCutPoints(start, cut);
        computeCutPoints(cut, end);
    }
    void CPPFImdlp::computeCutPointsOriginal(size_t start, size_t end)
    {
        precision_t cut;
        if (end - start < 2)
            return;
        cut = getCandidate(start, end);
        if (cut == -1)
            return;
        if (mdlp(start, cut, end)) {
            cutPoints.push_back((X[indices[cut]] + X[indices[cut - 1]]) / 2);
        }
        computeCutPointsOriginal(start, cut);
        computeCutPointsOriginal(cut, end);
    }
    void CPPFImdlp::computeCutPointsProposal()
    {
        precision_t xPrev, xCur, xPivot, cutPoint;
        int yPrev, yCur, yPivot;
        size_t idx, numElements, start;

        xCur = xPrev = X[indices[0]];
        yCur = yPrev = y[indices[0]];
        numElements = indices.size() - 1;
        idx = start = 0;
        while (idx < numElements) {
            xPivot = xCur;
            yPivot = yCur;
            // Read the same values and check class changes
            do {
                idx++;
                xCur = X[indices[idx]];
                yCur = y[indices[idx]];
                if (yCur != yPivot && xCur == xPivot) {
                    yPivot = -1;
                }
            }
            while (idx < numElements && xCur == xPivot);
            // Check if the class changed and there are more than 1 element
            if ((idx - start > 1) && (yPivot == -1 || yPrev != yCur) && mdlp(start, idx, indices.size())) {
                start = idx;
                cutPoint = (xPrev + xCur) / 2;
                cutPoints.push_back(cutPoint);
            }
            yPrev = yPivot;
            xPrev = xPivot;
        }
    }
    long int CPPFImdlp::getCandidate(size_t start, size_t end)
    {
        long int candidate = -1, elements = end - start;
        precision_t entropy_left, entropy_right, minEntropy = numeric_limits<precision_t>::max();
        for (auto idx = start + 1; idx < end; idx++) {
            // Cutpoints are always on boudndaries
            if (y[indices[idx]] == y[indices[idx - 1]])
                continue;
            entropy_left = precision_t(idx - start) / elements * metrics.entropy(start, idx);
            entropy_right = precision_t(end - idx) / elements * metrics.entropy(idx, end);
            if (entropy_left + entropy_right < minEntropy) {
                minEntropy = entropy_left + entropy_right;
                candidate = idx;
            }
        }
        return candidate;
    }
    bool CPPFImdlp::mdlp(size_t start, size_t cut, size_t end)
    {
        int k, k1, k2;
        precision_t ig, delta;
        precision_t ent, ent1, ent2;
        auto N = precision_t(end - start);
        if (N < 2) {
            return false;
        }
        k = metrics.computeNumClasses(start, end);
        k1 = metrics.computeNumClasses(start, cut);
        k2 = metrics.computeNumClasses(cut, end);
        ent = metrics.entropy(start, end);
        ent1 = metrics.entropy(start, cut);
        ent2 = metrics.entropy(cut, end);
        ig = metrics.informationGain(start, cut, end);
        delta = log2(pow(3, precision_t(k)) - 2) -
            (precision_t(k) * ent - precision_t(k1) * ent1 - precision_t(k2) * ent2);
        precision_t term = 1 / N * (log2(N - 1) + delta);
        return ig > term;
    }
    cutPoints_t CPPFImdlp::getCutPoints()
    {
        // Remove duplicates and sort
        cutPoints_t output(cutPoints.size());
        set<precision_t> s;
        unsigned size = cutPoints.size();
        for (unsigned i = 0; i < size; i++)
            s.insert(cutPoints[i]);
        output.assign(s.begin(), s.end());
        sort(output.begin(), output.end());
        return output;
    }
    // Argsort from https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
    indices_t CPPFImdlp::sortIndices(samples_t& X_)
    {
        indices_t idx(X_.size());
        iota(idx.begin(), idx.end(), 0);
        for (size_t i = 0; i < X_.size(); i++)
            sort(idx.begin(), idx.end(), [&X_](size_t i1, size_t i2)
                { return X_[i1] < X_[i2]; });
        return idx;
    }
}
