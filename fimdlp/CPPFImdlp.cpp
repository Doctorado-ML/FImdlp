#include "CPPFImdlp.h"
#include <numeric>
#include <iostream>
namespace CPPFImdlp
{
    CPPFImdlp::CPPFImdlp()
    {
    }
    CPPFImdlp::~CPPFImdlp()
    {
    }
    std::vector<double> CPPFImdlp::cutPoints(std::vector<float> &X, std::vector<int> &y)
    {
        std::vector<double> cutPts;
        double antx;
        // int anty;
        std::vector<size_t> indices = sortIndices(X);
        antx = X.at(indices[0]);
        // anty = y.at(indices[0]);
        for (auto index = indices.begin(); index != indices.end(); ++index)
        {
            // std::cout << X.at(*index) << " -> " << y.at(*index) << " // ";
            //  Definition 2 Cut points are always on boundaries
            // if (y.at(*index) != anty && antx < X.at(*index))
            //  Weka implementation
            if (antx < X.at(*index))
            {
                // std::cout << "* (" << X.at(*index) << ", " << antx << ") // ";
                cutPts.push_back((X.at(*index) + antx) / 2);
                // anty = y.at(*index);
            }
            antx = X.at(*index);
        }
        // std::cout << std::endl;
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
