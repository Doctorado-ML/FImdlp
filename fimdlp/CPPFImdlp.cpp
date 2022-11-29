#include "CPPFImdlp.h"
#include <numeric>
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include "Metrics.h"
namespace CPPFImdlp
{
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
    std::vector<float> CPPFImdlp::cutPoints(std::vector<float> &X, std::vector<int> &y)
    {
        std::vector<float> cutPts;
        std::vector<size_t> cutIdx;
        float xPrev, cutPoint, curx;
        int yPrev, cury;
        size_t idxPrev, idx;
        bool first = true;
        std::vector<size_t> indices = sortIndices(X);
        xPrev = X.at(indices.at(0));
        yPrev = y.at(indices.at(0));
        idxPrev = indices.at(0);
        idx = 0;
        while (idx < indices.size() - 1)
        {
            if (first)
            {
                first = false;
                curx = X.at(indices.at(idx));
                cury = y.at(indices.at(idx));
            }
            if (debug)
                printf("<idx=%lu -> (%3.1f, %d) Prev(%3.1f, %d)\n", idx, curx, cury, xPrev, yPrev);
            // Read the same values and check class changes
            while (idx < indices.size() - 1 && curx == xPrev)
            {
                idx++;
                curx = X.at(indices.at(idx));
                cury = y.at(indices.at(idx));
                if (cury != yPrev && curx == xPrev)
                {
                    yPrev = -1;
                }
                if (debug)
                    printf(">idx=%lu -> (%3.1f, %d) Prev(%3.1f, %d)\n", idx, curx, cury, xPrev, yPrev);
            }
            if (yPrev == -1 || yPrev != cury)
            {
                cutPoint = (xPrev + curx) / 2;
                printf("Cutpoint (%3.1f, %d) -> (%3.1f, %d) = %3.1f", xPrev, yPrev, curx, cury, cutPoint);
                cutPts.push_back(cutPoint);
                cutIdx.push_back(idxPrev);
            }
            yPrev = cury;
            xPrev = curx;
            idxPrev = indices.at(idx);
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
