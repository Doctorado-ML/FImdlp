#include "CPPFImdlp.h"
#include <numeric>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include "Metrics.h"
namespace mdlp {
    CPPFImdlp::CPPFImdlp() : debug(false), precision(6)
    {
        divider = pow(10, precision);
    }
    CPPFImdlp::CPPFImdlp(int precision, bool debug) : debug(debug), precision(precision)
    {
        divider = pow(10, precision);
    }
    CPPFImdlp::~CPPFImdlp()
    {
    }
    std::vector<CutPoint_t> CPPFImdlp::getCutPoints()
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

        computeCutPoints();
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
    bool CPPFImdlp::evaluateCutPoint(CutPoint_t rest, CutPoint_t candidate)
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
        std::cout << candidate
            std::cout << "k=" << k << " k1=" << k1 << " k2=" << k2 << " ent=" << ent << " ent1=" << ent1 << " ent2=" << ent2 << std::endl;
        std::cout << "ig=" << ig << " delta=" << delta << " N " << N << " term " << term << std::endl;
        return (ig > term);
    }
    void CPPFImdlp::filterCutPoints()
    {
        std::vector<CutPoint_t> filtered;
        CutPoint_t rest;
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
            } else {
                std::cout << "Rejected" << std::endl;
                lastReject = true;
            }
        }
        if (!first)
            filtered.back().toValue = std::numeric_limits<float>::max();
        else {
            filtered.push_back(rest);
        }

        cutPoints = filtered;
    }
    void CPPFImdlp::computeCutPoints()
    {

        std::vector<CutPoint_t> cutPts;
        CutPoint_t cutPoint;
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
    void CPPFImdlp::computeCutPointsAnt()
    {
        samples cutPts;
        labels cutIdx;
        float xPrev, cutPoint;
        int yPrev;
        size_t idxPrev;
        xPrev = X.at(indices[0]);
        yPrev = y.at(indices[0]);
        idxPrev = indices[0];
        if (debug) {
            std::cout << "Entropy: " << Metrics::entropy(y, indices, 0, y.size(), Metrics::numClasses(y, indices, 0, indices.size())) << std::endl;
        }
        for (auto index = indices.begin(); index != indices.end(); ++index) {
            //  Definition 2 Cut points are always on boundaries
            if (y.at(*index) != yPrev && xPrev < X.at(*index)) {
                cutPoint = round(divider * (X.at(*index) + xPrev) / 2) / divider;
                if (debug) {
                    std::cout << "Cut point: " << (xPrev + X.at(*index)) / 2 << " //";
                    std::cout << X.at(*index) << " -> " << y.at(*index) << " yPrev= " << yPrev;
                    std::cout << "* (" << X.at(*index) << ", " << xPrev << ")="
                        << ((X.at(*index) + xPrev) / 2) << "idxPrev"
                        << idxPrev << std::endl;
                }
                cutPts.push_back(cutPoint);
                cutIdx.push_back(idxPrev);
            }
            xPrev = X.at(*index);
            yPrev = y.at(*index);
            idxPrev = *index;
        }
        // cutPoints = cutPts;
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
