#include "CPPFImdlp.h"
#include <numeric>
#include <iostream>
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
        float antx, cutPoint;
        int anty;
        std::vector<size_t> indices = sortIndices(X);
        antx = X.at(indices[0]);
        anty = y.at(indices[0]);
        for (auto index = indices.begin(); index != indices.end(); ++index)
        {
            // std::cout << X.at(*index) << " -> " << y.at(*index) << " // ";
            //  Definition 2 Cut points are always on boundaries
            if (y.at(*index) != anty && antx < X.at(*index))
            //  Weka implementation
            // if (antx < X.at(*index))
            {
                cutPoint = round((X.at(*index) + antx) / 2 * divider) / divider;
                if (debug)
                {
                    std::cout << "Cut point: " << (antx + X.at(*index)) / 2 << " //";
                    std::cout << X.at(*index) << " -> " << y.at(*index) << " anty= " << anty;
                    std::cout << "* (" << X.at(*index) << ", " << antx << ")=" << ((X.at(*index) + antx) / 2) << std::endl;
                }
                cutPts.push_back(cutPoint);
            }
            antx = X.at(*index);
            anty = y.at(*index);
        }
        return cutPts;
    }
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
