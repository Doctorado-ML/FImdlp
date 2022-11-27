#include "CPPFImdlp.h"
namespace CPPFImdlp
{
    CPPFImdlp::CPPFImdlp()
    {
    }
    CPPFImdlp::~CPPFImdlp()
    {
    }
    std::vector<float> CPPFImdlp::cutPoints(std::vector<int> &X, std::vector<int> &y)
    {
        std::vector<float> cutPts;
        int i, ant = X.at(0);
        int n = X.size();
        for (i = 1; i < n; i++)
        {
            if (X.at(i) != ant)
            {
                cutPts.push_back(float(X.at(i) + ant) / 2);
                ant = X.at(i);
            }
        }
        return cutPts;
    }
}