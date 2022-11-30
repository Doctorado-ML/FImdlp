#include "CPPFImdlp.h"
#include <numeric>
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <algorithm>
#include "Metrics.h"
namespace CPPFImdlp
{
    std::ostream &operator<<(std::ostream &os, const CutPointBody &cut)
    {
        os << "(" << cut.start << ", " << cut.end << ") -> (" << cut.fromValue << ",  " << cut.toValue << "]";
        return os;
    }
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
    void CPPFImdlp::debugPoints(std::vector<float> &X, std::vector<int> &y)
    {
        std::cout << "+++++++++++++++++++++++" << std::endl;
        // for (auto i : sortIndices(X))
        std::vector<size_t> indices = sortIndices(X);
        for (size_t i = 0; i < indices.size(); i++)
        {
            printf("(%3lu, %3lu) -> (%3.1f, %d)\n", i, indices[i], X[indices[i]], y[indices[i]]);
        }
        std::cout << "+++++++++++++++++++++++" << std::endl;
        for (auto item : cutPoints(X, y))
        {
            std::cout << item << "  X[" << item.end << "]=" << X[item.end] << std::endl;
        }
    }
    std::vector<CutPointBody> CPPFImdlp::cutPoints(std::vector<float> &X, std::vector<int> &y)
    {

        std::vector<CutPointBody> cutPts;
        CutPointBody cutPoint;
        std::vector<size_t> cutIdx;
        float xPrev, xCur, xPivot;
        int yPrev, yCur, yPivot;
        size_t idxPrev, idxPivot, idx, numElements, start;
        std::vector<size_t> indices = sortIndices(X);
        xCur = xPrev = X[indices[0]];
        yCur = yPrev = y[indices[0]];
        numElements = indices.size() - 1;
        idxPrev = indices[0];
        idx = start = 0;
        bool firstCutPoint = true;
        if (debug)
            printf("*idx=%lu -> (-1, -1) Prev(%3.1f, %d) Elementos: %lu\n", idx, xCur, yCur, numElements);
        while (idx < numElements)
        {
            xPivot = xCur;
            yPivot = yCur;
            idxPivot = indices[idx];
            if (debug)
                printf("<idx=%lu -> Prev(%3.1f, %d) Pivot(%3.1f, %d) Cur(%3.1f, %d) \n", idx, xPrev, yPrev, xPivot, yPivot, xCur, yCur);
            // Read the same values and check class changes
            do
            {
                idx++;
                xCur = X[indices[idx]];
                yCur = y[indices[idx]];
                if (yCur != yPivot && xCur == xPivot)
                {
                    yPivot = -1;
                }
                if (debug)
                    printf(">idx=%lu -> Prev(%3.1f, %d) Pivot(%3.1f, %d) Cur(%3.1f, %d) \n", idx, xPrev, yPrev, xPivot, yPivot, xCur, yCur);
            } while (idx < numElements && xCur == xPivot);
            if (yPivot == -1 || yPrev != yCur)
            {
                cutPoint.start = start;
                cutPoint.end = idxPrev;
                start = idx;
                cutPoint.fromValue = firstCutPoint ? std::numeric_limits<float>::lowest() : cutPts.back().toValue;
                cutPoint.toValue = (xPrev + xCur) / 2;
                firstCutPoint = false;
                if (debug)
                {
                    printf("Cutpoint idx=%lu Cur(%3.1f, %d) Prev(%3.1f, %d) Pivot(%3.1f, %d) = (%3.1g, %3.1g] \n", idx, xCur, yCur, xPrev, yPrev, xPivot, yPivot, cutPoint.fromValue, cutPoint.toValue);
                }
                cutPts.push_back(cutPoint);
                cutIdx.push_back(idxPrev);
            }
            yPrev = yPivot;
            xPrev = xPivot;
            idxPrev = indices[idxPivot];
        }
        if (idxPrev >= numElements)
        {
            cutPoint.start = start;
            cutPoint.end = numElements;
            cutPoint.fromValue = firstCutPoint ? std::numeric_limits<float>::lowest() : cutPts.back().toValue;
            cutPoint.toValue = std::numeric_limits<float>::max();
            if (debug)
                printf("Final Cutpoint idx=%lu Cur(%3.1f, %d) Prev(%3.1f, %d) Pivot(%3.1f, %d) = (%3.1g, %3.1g] \n", idx, xCur, yCur, xPrev, yPrev, xPivot, yPivot, cutPoint.fromValue, cutPoint.toValue);
            cutPts.push_back(cutPoint);
            cutIdx.push_back(idxPrev);
        }
        return cutPts;
    }
    std::vector<float> CPPFImdlp::cutPointsAnt(std::vector<float> &X, std::vector<int> &y)
    {
        std::vector<float> cutPts;
        std::vector<int> cutIdx;
        float xPrev, cutPoint;
        int yPrev;
        size_t idxPrev;
        std::vector<size_t> indices = sortIndices(X);
        xPrev = X.at(indices[0]);
        yPrev = y.at(indices[0]);
        idxPrev = indices[0];
        if (debug)
        {
            std::cout << "Entropy: " << Metrics::entropy(y, indices, 0, y.size(), Metrics::numClasses(y, indices, 0, indices.size())) << std::endl;
        }
        for (auto index = indices.begin(); index != indices.end(); ++index)
        {
            //  Definition 2 Cut points are always on boundaries
            if (y.at(*index) != yPrev && xPrev < X.at(*index))
            {
                cutPoint = round(divider * (X.at(*index) + xPrev) / 2) / divider;
                if (debug)
                {
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
        return cutPts;
    }
    // Argsort from https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
    std::vector<size_t> CPPFImdlp::sortIndices(std::vector<float> &X)
    {
        std::vector<size_t> idx(X.size());
        std::iota(idx.begin(), idx.end(), 0);
        for (std::size_t i = 0; i < X.size(); i++)
            stable_sort(idx.begin(), idx.end(), [&X](size_t i1, size_t i2)
                        { return X[i1] < X[i2]; });
        return idx;
    }
}
