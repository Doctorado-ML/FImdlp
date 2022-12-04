#include "CPPFImdlp.h"
#include <numeric>
#include <iostream>
#include <algorithm>
#include "Metrics.h"
namespace mdlp {
    std::ostream& operator << (std::ostream& os, const cutPoint_t& cut)
    {
        os << cut.classNumber << " -> (" << cut.start << ", " << cut.end <<
            ") - (" << cut.fromValue << ", " << cut.toValue << ")  "
            << std::endl;
        return os;

    }
    CPPFImdlp::CPPFImdlp() : proposed(true), precision(6), debug(false)
    {
        divider = pow(10, precision);
        numClasses = 0;
    }
    CPPFImdlp::CPPFImdlp(bool proposed, int precision, bool debug) : proposed(proposed), precision(precision), debug(debug)
    {
        divider = pow(10, precision);
        numClasses = 0;
    }
    CPPFImdlp::~CPPFImdlp()
        = default;
    std::vector<cutPoint_t> CPPFImdlp::getCutPoints()
    {
        return cutPoints;
    }
    labels CPPFImdlp::getDiscretizedValues()
    {
        return xDiscretized;
    }
    CPPFImdlp& CPPFImdlp::fit(samples& X_, labels& y_)
    {
        X = X_;
        y = y_;
        if (X.size() != y.size()) {
            std::cerr << "X and y must have the same size" << std::endl;
            return *this;
        }
        if (X.size() == 0) {
            std::cerr << "X and y must have at least one element" << std::endl;
            return *this;
        }
        this->indices = sortIndices(X_);
        this->xDiscretized = labels(X.size(), -1);
        this->numClasses = Metrics::numClasses(y, indices, 0, X.size());

        if (proposed) {
            computeCutPointsProposed();
        } else {
            computeCutPointsOriginal();
        }
        filterCutPoints();
        applyCutPoints();
        return *this;
    }
    labels& CPPFImdlp::transform(samples& X_)
    {
        indices_t indices_transform = sortIndices(X_);
        applyCutPoints();
        return xDiscretized;
    }
    void CPPFImdlp::applyCutPoints()
    {
        for (auto cut : cutPoints) {
            for (size_t i = cut.start; i < cut.end; i++) {
                xDiscretized[indices[i]] = cut.classNumber;
            }
        }
    }
    bool CPPFImdlp::evaluateCutPoint(cutPoint_t rest, cutPoint_t candidate)
    {
        int k, k1, k2;
        float ig, delta;
        float ent, ent1, ent2;
        auto N = float(rest.end - rest.start);
        if (N < 2) {
            return false;
        }
        k = Metrics::numClasses(y, indices, rest.start, rest.end);
        k1 = Metrics::numClasses(y, indices, rest.start, candidate.end);
        k2 = Metrics::numClasses(y, indices, candidate.end, rest.end);
        ent = Metrics::entropy(y, indices, rest.start, rest.end, numClasses);
        ent1 = Metrics::entropy(y, indices, rest.start, candidate.end, numClasses);
        ent2 = Metrics::entropy(y, indices, candidate.end, rest.end, numClasses);
        ig = Metrics::informationGain(y, indices, rest.start, rest.end, candidate.end, numClasses);
        delta = log2(pow(3, float(k)) - 2) - (float(k) * ent - float(k1) * ent1 - float(k2) * ent2);
        float term = 1 / N * (log2(N - 1) + delta);
        if (debug) {
            std::cout << "Rest: " << rest;
            std::cout << "Candidate: " << candidate;
            std::cout << "k=" << k << " k1=" << k1 << " k2=" << k2 << " ent=" << ent << " ent1=" << ent1 << " ent2=" << ent2 << std::endl;
            std::cout << "ig=" << ig << " delta=" << delta << " N " << N << " term " << term << std::endl;
        }
        return (ig > term);
    }
    void CPPFImdlp::filterCutPoints()
    {
        cutPoints_t filtered;
        cutPoint_t rest, item;
        int classNumber = 0;

        rest.start = 0;
        rest.end = X.size();
        rest.fromValue = std::numeric_limits<float>::lowest();
        rest.toValue = std::numeric_limits<float>::max();
        rest.classNumber = classNumber;
        bool first = true;
        for (size_t index = 0; index < size_t(cutPoints.size()); index++) {
            item = cutPoints[index];
            if (evaluateCutPoint(rest, item)) {
                if (debug)
                    std::cout << "Accepted: " << item << std::endl;
                //Assign class number to the interval (cutpoint)
                item.classNumber = classNumber++;
                filtered.push_back(item);
                first = false;
                rest.start = item.end;
            } else {
                if (debug)
                    std::cout << "Rejected: " << item << std::endl;
                if (index != size_t(cutPoints.size()) - 1) {
                    // Try to merge the rejected cutpoint with the next one
                    if (first) {
                        cutPoints[index + 1].fromValue = std::numeric_limits<float>::lowest();
                        cutPoints[index + 1].start = indices[0];
                    } else {
                        cutPoints[index + 1].fromValue = item.fromValue;
                        cutPoints[index + 1].start = item.start;
                    }
                }
            }
        }
        if (!first) {
            filtered.back().toValue = std::numeric_limits<float>::max();
            filtered.back().end = X.size() - 1;
        } else {
            filtered.push_back(rest);
        }
        cutPoints = filtered;
    }
    void CPPFImdlp::computeCutPointsProposed()
    {
        cutPoints_t cutPts;
        cutPoint_t cutPoint;
        float xPrev, xCur, xPivot;
        int yPrev, yCur, yPivot;
        size_t idx, numElements, start;

        xCur = xPrev = X[indices[0]];
        yCur = yPrev = y[indices[0]];
        numElements = indices.size() - 1;
        idx = start = 0;
        bool firstCutPoint = true;
        if (debug)
            printf("*idx=%lu -> (-1, -1) Prev(%3.1f, %d) Elementos: %lu\n", idx, xCur, yCur, numElements);
        while (idx < numElements) {
            xPivot = xCur;
            yPivot = yCur;
            if (debug)
                printf("<idx=%lu -> Prev(%3.1f, %d) Pivot(%3.1f, %d) Cur(%3.1f, %d) \n", idx, xPrev, yPrev, xPivot, yPivot, xCur, yCur);
            // Read the same values and check class changes
            do {
                idx++;
                xCur = X[indices[idx]];
                yCur = y[indices[idx]];
                if (yCur != yPivot && xCur == xPivot) {
                    yPivot = -1;
                }
                if (debug)
                    printf(">idx=%lu -> Prev(%3.1f, %d) Pivot(%3.1f, %d) Cur(%3.1f, %d) \n", idx, xPrev, yPrev, xPivot, yPivot, xCur, yCur);
            }
            while (idx < numElements && xCur == xPivot);
            if (yPivot == -1 || yPrev != yCur) {
                cutPoint.start = start;
                cutPoint.end = idx;
                start = idx;
                cutPoint.fromValue = firstCutPoint ? std::numeric_limits<float>::lowest() : cutPts.back().toValue;
                cutPoint.toValue = (xPrev + xCur) / 2;
                cutPoint.classNumber = -1;
                firstCutPoint = false;
                if (debug) {
                    printf("Cutpoint idx=%lu Cur(%3.1f, %d) Prev(%3.1f, %d) Pivot(%3.1f, %d) = (%3.1g, %3.1g] \n", idx, xCur, yCur, xPrev, yPrev, xPivot, yPivot, cutPoint.fromValue, cutPoint.toValue);
                }
                cutPts.push_back(cutPoint);
            }
            yPrev = yPivot;
            xPrev = xPivot;
        }
        if (idx == numElements) {
            cutPoint.start = start;
            cutPoint.end = numElements + 1;
            cutPoint.fromValue = firstCutPoint ? std::numeric_limits<float>::lowest() : cutPts.back().toValue;
            cutPoint.toValue = std::numeric_limits<float>::max();
            cutPoint.classNumber = -1;
            if (debug)
                printf("Final Cutpoint idx=%lu Cur(%3.1f, %d) Prev(%3.1f, %d) Pivot(%3.1f, %d) = (%3.1g, %3.1g] \n", idx, xCur, yCur, xPrev, yPrev, xPivot, yPivot, cutPoint.fromValue, cutPoint.toValue);
            cutPts.push_back(cutPoint);
        }
        if (debug)
            for (auto cutPt : cutPts)
                std::cout << "Proposed: Cut point: " << cutPt;
        cutPoints = cutPts;
    }
    void CPPFImdlp::computeCutPointsOriginal()
    {
        cutPoints_t cutPts;
        cutPoint_t cutPoint;
        float xPrev;
        int yPrev;
        bool first = true;
        // idxPrev is the index of the init instance of the cutPoint
        size_t index, idxPrev = 0, idx = indices[0];
        xPrev = X[idx];
        yPrev = y[idx];
        for (index = 0; index < size_t(indices.size()) - 1; index++) {
            idx = indices[index];
            //  Definition 2 Cut points are always on boundaries
            if (y[idx] != yPrev && xPrev < X[idx]) {
                if (first) {
                    first = false;
                    cutPoint.fromValue = std::numeric_limits<float>::lowest();
                } else {
                    cutPoint.fromValue = cutPts.back().toValue;
                }
                cutPoint.start = idxPrev;
                cutPoint.end = index;
                cutPoint.classNumber = -1;
                cutPoint.toValue = round(divider * (X[idx] + xPrev) / 2) / divider;
                idxPrev = index;
                cutPts.push_back(cutPoint);
            }
            xPrev = X[idx];
            yPrev = y[idx];
        }
        if (first) {
            cutPoint.start = 0;
            cutPoint.classNumber = -1;
            cutPoint.fromValue = std::numeric_limits<float>::lowest();
            cutPoint.toValue = std::numeric_limits<float>::max();
            cutPts.push_back(cutPoint);
        } else
            cutPts.back().toValue = std::numeric_limits<float>::max();
        cutPts.back().end = X.size();
        if (debug)
            for (auto cutPt : cutPts)
                std::cout << "Original: Cut point: " << cutPt;
        cutPoints = cutPts;
    }
    // Argsort from https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
    indices_t CPPFImdlp::sortIndices(samples& X_)
    {
        indices_t idx(X_.size());
        std::iota(idx.begin(), idx.end(), 0);
        for (std::size_t i = 0; i < X_.size(); i++)
            stable_sort(idx.begin(), idx.end(), [&X_](size_t i1, size_t i2)
                { return X_[i1] < X_[i2]; });
        return idx;
    }
    void CPPFImdlp::setCutPoints(cutPoints_t cutPoints_)
    {
        cutPoints = cutPoints_;
    }
    indices_t CPPFImdlp::getIndices()
    {
        return indices;
    }
}
