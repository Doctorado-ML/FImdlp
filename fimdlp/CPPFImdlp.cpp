#include "CPPFImdlp.h"
#include <numeric>
#include <iostream>
#include <iomanip>
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
    }
    CPPFImdlp::CPPFImdlp(bool proposed, int precision, bool debug) : proposed(proposed), precision(precision), debug(debug)
    {
        divider = pow(10, precision);
    }
    CPPFImdlp::~CPPFImdlp()
    {
    }
    std::vector<cutPoint_t> CPPFImdlp::getCutPoints()
    {
        return cutPoints;
    }
    labels CPPFImdlp::getDiscretizedValues()
    {
        return xDiscretized;
    }
    void CPPFImdlp::fit(samples& X, labels& y)
    {
        this->X = X;
        this->y = y;
        this->indices = sortIndices(X);
        this->xDiscretized = labels(X.size(), -1);
        this->numClasses = Metrics::numClasses(y, indices, 0, X.size());

        if (proposed) {
            computeCutPointsProposed();
        } else {
            computeCutPointsOriginal();
        }
        filterCutPoints();
        applyCutPoints();
    }
    labels& CPPFImdlp::transform(samples& X)
    {
        indices_t indices_transform = sortIndices(X);
        applyCutPoints();
        return xDiscretized;
    }
    void CPPFImdlp::debugPoints(samples& X, labels& y)
    {
        std::cout << "+++++++++++++++++++++++" << std::endl;
        // for (auto i : sortIndices(X))
        indices_t indices = sortIndices(X);
        for (size_t i = 0; i < indices.size(); i++) {
            printf("(%3lu, %3lu) -> (%3.1f, %d)\n", i, indices[i], X[indices[i]], y[indices[i]]);
        }
        std::cout << "+++++++++++++++++++++++" << std::endl;
        fit(X, y);
        for (auto item : cutPoints) {
            std::cout << item.start << "  X[" << item.end << "]=" << X[item.end] << std::endl;
        }
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
        float N = float(rest.end - rest.start);
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
        delta = log2(pow(3, k) - 2) - (k * ent - k1 * ent1 - k2 * ent2);
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
        cutPoint_t rest;
        int classNumber = 0;

        rest.start = 0;
        rest.end = X.size();
        rest.fromValue = std::numeric_limits<float>::lowest();
        rest.toValue = std::numeric_limits<float>::max();
        rest.classNumber = classNumber;
        bool lastReject = false, first = true;
        for (auto item : cutPoints) {
            if (evaluateCutPoint(rest, item)) {
                std::cout << "Accepted" << std::endl;
                if (lastReject) {
                    if (first) {
                        item.fromValue = std::numeric_limits<float>::lowest();
                        item.start = indices[0];
                    } else {
                        item.fromValue = filtered.back().toValue;
                        item.start = filtered.back().end;
                    }
                }
                //Assign class number to the interval (cutpoint)
                item.classNumber = classNumber++;
                filtered.push_back(item);
                first = false;
                rest.start = item.end;
            } else {
                std::cout << "Rejected" << std::endl;
                lastReject = true;
            }
        }
        if (!first) {
            filtered.back().toValue = std::numeric_limits<float>::max();
            filtered.back().end = X.size();
        } else {
            filtered.push_back(rest);
        }

        cutPoints = filtered;
    }
    void CPPFImdlp::computeCutPointsProposed()
    {
        cutPoints_t cutPts;
        cutPoint_t cutPoint;
        indices_t cutIdx;
        float xPrev, xCur, xPivot;
        int yPrev, yCur, yPivot;
        size_t idxPrev, idxPivot, idx, numElements, start;

        xCur = xPrev = X[indices[0]];
        yCur = yPrev = y[indices[0]];
        numElements = indices.size() - 1;
        idxPrev = indices[0];
        idx = start = 0;
        bool firstCutPoint = true;
        if (debug)
            printf("*idx=%lu -> (-1, -1) Prev(%3.1f, %d) Elementos: %lu\n", idx, xCur, yCur, numElements);
        while (idx < numElements) {
            xPivot = xCur;
            yPivot = yCur;
            idxPivot = indices[idx];
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
                cutPoint.end = idx - 1;
                start = idx;
                cutPoint.fromValue = firstCutPoint ? std::numeric_limits<float>::lowest() : cutPts.back().toValue;
                cutPoint.toValue = (xPrev + xCur) / 2;
                cutPoint.classNumber = -1;
                firstCutPoint = false;
                if (debug) {
                    printf("Cutpoint idx=%lu Cur(%3.1f, %d) Prev(%3.1f, %d) Pivot(%3.1f, %d) = (%3.1g, %3.1g] \n", idx, xCur, yCur, xPrev, yPrev, xPivot, yPivot, cutPoint.fromValue, cutPoint.toValue);
                }
                cutPts.push_back(cutPoint);
                cutIdx.push_back(idxPrev);
            }
            yPrev = yPivot;
            xPrev = xPivot;
            idxPrev = indices[idxPivot];
        }
        if (idx == numElements) {
            cutPoint.start = start;
            cutPoint.end = numElements;
            cutPoint.fromValue = firstCutPoint ? std::numeric_limits<float>::lowest() : cutPts.back().toValue;
            cutPoint.toValue = std::numeric_limits<float>::max();
            cutPoint.classNumber = -1;
            if (debug)
                printf("Final Cutpoint idx=%lu Cur(%3.1f, %d) Prev(%3.1f, %d) Pivot(%3.1f, %d) = (%3.1g, %3.1g] \n", idx, xCur, yCur, xPrev, yPrev, xPivot, yPivot, cutPoint.fromValue, cutPoint.toValue);
            cutPts.push_back(cutPoint);
            cutIdx.push_back(idxPrev);
        }
        cutPoints = cutPts;
    }
    void CPPFImdlp::computeCutPointsOriginal()
    {
        cutPoints_t cutPts;
        cutPoint_t cutPoint;
        float xPrev = std::numeric_limits<float>::lowest();
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
                if (debug) {
                    std::cout << "Cut point: " << cutPoint << " //";
                    std::cout << X[idx] << " -> " << y[idx] << " yPrev= "
                        << yPrev << idxPrev << std::endl;
                }
                idxPrev = index;
                cutPts.push_back(cutPoint);
            }
            xPrev = X[idx];
            yPrev = y[idx];
        }
        std::cout << "Came to here" << first << std::endl;
        if (first) {
            cutPoint.start = 0;
            cutPoint.classNumber = -1;
            cutPoint.fromValue = std::numeric_limits<float>::lowest();
            cutPoint.toValue = std::numeric_limits<float>::max();
            cutPoints.push_back(cutPoint);
        } else
            cutPts.back().toValue = std::numeric_limits<float>::max();
        cutPts.back().end = X.size();
        if (debug)
            for (auto cutPoint : cutPts)
                std::cout << "Cut point: " << cutPoint << std::endl;
        cutPoints = cutPts;
    }
    // Argsort from https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
    indices_t CPPFImdlp::sortIndices(samples& X)
    {
        indices_t idx(X.size());
        std::iota(idx.begin(), idx.end(), 0);
        for (std::size_t i = 0; i < X.size(); i++)
            stable_sort(idx.begin(), idx.end(), [&X](size_t i1, size_t i2)
                { return X[i1] < X[i2]; });
        return idx;
    }
}
