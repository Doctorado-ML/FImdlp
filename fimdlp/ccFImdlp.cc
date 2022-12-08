#include "ccFImdlp.h"
#include <numeric>
#include <iostream>
#include <algorithm>
#include <set>
#include "ccMetrics.h"

namespace mdlp {
    CPPFImdlp::CPPFImdlp(): proposal(true), precision(6), debug(false), divider(pow(10, precision)), indices(indices_t()), y(labels()), metrics(Metrics(y, indices))
    {
    }
    CPPFImdlp::CPPFImdlp(bool proposal, int precision, bool debug): proposal(proposal), precision(precision), debug(debug), divider(pow(10, precision)), indices(indices_t()), y(labels()), metrics(Metrics(y, indices))
    {
    }
    CPPFImdlp::~CPPFImdlp()
        = default;

    CPPFImdlp& CPPFImdlp::fitx(samples& X_, labels& y_)
    {
        X = X_;
        y = y_;
        if (X.size() != y.size()) {
            throw invalid_argument("X and y must have the same size");
        }
        if (X.size() == 0 || y.size() == 0) {
            throw invalid_argument("X and y must have at least one element");
        }
        indices = sortIndices(X_);
        metrics.setData(y, indices);
        computeCutPointsRecursive(0, X.size());
        //simulateCutPointsRecursive();
        return *this;
    }
    void CPPFImdlp::simulateCutPointsRecursive()
    {
        cutPoints_t jobs = cutPoints_t();
        jobs.push_back(cutPoint_t({ 0, X.size() }));
        while (jobs.size() > 0) {
            auto interval = jobs.back();
            jobs.pop_back();
            //cout << "start: " << interval.start << " end: " << interval.end << endl;
            auto cut = getCandidateSimulate(interval.start, interval.end);
            if (cut == -1 || !mdlp(interval.start, cut, interval.end)) {
                if (interval.start != 0)
                    xCutPoints.push_back(xcutPoint_t({ interval.start, (X[indices[interval.start]] + X[indices[interval.start - 1]]) / 2 }));
                if (interval.end != X.size())
                    xCutPoints.push_back(xcutPoint_t({ interval.end, (X[indices[interval.end]] + X[indices[interval.end - 1]]) / 2 }));
                continue;
            }
            jobs.push_back(cutPoint_t({ interval.start, size_t(cut) }));
            jobs.push_back(cutPoint_t({ size_t(cut), interval.end }));

        }
    }
    void CPPFImdlp::computeCutPointsRecursive(size_t start, size_t end)
    {
        xcutPoint_t cut;
        //cout << "start: " << start << " end: " << end << endl;
        if (end - start < 2)
            return;
        cut = getCandidate(start, end);
        if (cut.value == -1 || !mdlp(start, cut.index, end)) {
            // cut.value == -1 means that there is no candidate in the interval
            // that enhances the information gain
            //cout << "¡Ding! " << cut.value << " " << cut.index << endl;
            if (start != 0)
                xCutPoints.push_back(xcutPoint_t({ start, (X[indices[start]] + X[indices[start - 1]]) / 2 }));
            if (end != X.size())
                xCutPoints.push_back(xcutPoint_t({ end, (X[indices[end]] + X[indices[end - 1]]) / 2 }));
            return;
        }
        computeCutPointsRecursive(start, cut.index);
        computeCutPointsRecursive(cut.index, end);
    }
    xcutPoint_t CPPFImdlp::getCandidate(size_t start, size_t end)
    {
        xcutPoint_t candidate;
        int elements = end - start;
        candidate.value = -1;
        candidate.index = -1;
        float entropy_left, entropy_right, minEntropy = numeric_limits<float>::max();
        for (auto idx = start + 1; idx < end; idx++) {
            if (y[indices[idx]] == y[indices[idx - 1]])
                continue;
            entropy_left = float(idx - start) / elements * metrics.entropy(start, idx);
            entropy_right = float(end - idx) / elements * metrics.entropy(idx, end);
            if (entropy_left + entropy_right < minEntropy) {
                minEntropy = entropy_left + entropy_right;
                candidate.value = (X[indices[idx]] + X[indices[idx - 1]]) / 2;
                candidate.index = idx;
            }
        }
        return candidate;
    }
    int CPPFImdlp::getCandidateSimulate(size_t start, size_t end)
    {
        int candidate = -1;
        int elements = end - start;
        float entropy_left, entropy_right, minEntropy = numeric_limits<float>::max();
        for (auto idx = start + 1; idx < end; idx++) {
            if (y[indices[idx]] == y[indices[idx - 1]])
                continue;
            entropy_left = float(idx - start) / elements * metrics.entropy(start, idx);
            entropy_right = float(end - idx) / elements * metrics.entropy(idx, end);
            if (minEntropy > entropy_left + entropy_right) {
                minEntropy = entropy_left + entropy_right;
                candidate = idx;
            }
        }
        return candidate;
    }
    bool CPPFImdlp::mdlp(size_t start, size_t cut, size_t end)
    {
        int k, k1, k2;
        float ig, delta;
        float ent, ent1, ent2;
        auto N = float(end - start);
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
        delta = log2(pow(3, float(k)) - 2) - (float(k) * ent - float(k1) * ent1 - float(k2) * ent2);
        float term = 1 / N * (log2(N - 1) + delta);
        if (debug) {
            cout << "start: " << start << " cut: " << cut << " end: " << end << endl;
            cout << "k=" << k << " k1=" << k1 << " k2=" << k2 << " ent=" << ent << " ent1=" << ent1 << " ent2=" << ent2 << endl;
            cout << "ig=" << ig << " delta=" << delta << " N " << N << " term " << term << endl;
        }
        return ig > term;
    }
    samples CPPFImdlp::getCutPointsx()
    {
        // Remove duplicates and sort
        samples output(xCutPoints.size());
        set<float> s;
        unsigned size = xCutPoints.size();
        for (unsigned i = 0; i < size; i++)
            s.insert(xCutPoints[i].value);
        output.assign(s.begin(), s.end());
        sort(output.begin(), output.end());
        return output;
    }
    // Argsort from https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
    indices_t CPPFImdlp::sortIndices(samples& X_)
    {
        indices_t idx(X_.size());
        iota(idx.begin(), idx.end(), 0);
        for (size_t i = 0; i < X_.size(); i++)
            sort(idx.begin(), idx.end(), [&X_](size_t i1, size_t i2)
                { return X_[i1] < X_[i2]; });
        return idx;
    }
}
