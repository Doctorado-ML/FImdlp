#include "Metrics.h"
namespace CPPFImdlp
{
    Metrics::Metrics()
    {
    }
    float Metrics::entropy(std::vector<int> &y, int start, int end, int nClasses)
    {
        float entropy = 0;
        int nElements = end - start;
        std::vector<int>
            counts(nClasses, 0);
        for (auto i = start; i < end; i++)
        {
            counts[y[i]]++;
        }
        for (auto i = 0; i < nClasses; i++)
        {
            if (counts[i] > 0)
            {
                float p = (float)counts[i] / nElements;
                entropy -= p * log2(p);
            }
        }
        return entropy;
    }
    int Metrics::numClasses(std::vector<int> &y)
    {
        int nClasses = 1;
        int yAnt = y.at(0);
        for (auto i = y.begin(); i != y.end(); ++i)
        {
            if (*i != yAnt)
            {
                nClasses++;
            }
        }
        return nClasses;
    }
}
