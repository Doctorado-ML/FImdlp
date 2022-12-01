#include "Metrics.h"
#include <set>
namespace mdlp
{
    Metrics::Metrics()
    {
    }
    int Metrics::numClasses(std::vector<int> &y, std::vector<size_t> indices, size_t start, size_t end)
    {
        std::set<int> numClasses;
        for (auto i = start; i < end; ++i)
        {
            numClasses.insert(y[indices[i]]);
        }
        return numClasses.size();
    }
    float Metrics::entropy(std::vector<int> &y, std::vector<size_t> &indices, size_t start, size_t end, int nClasses)
    {
        float entropy = 0;
        int nElements = 0;
        std::vector<int> counts(nClasses + 1, 0);
        for (auto i = &indices[start]; i != &indices[end]; ++i)
        {
            counts[y[*i]]++;
            nElements++;
        }
        for (auto count : counts)
        {
            if (count > 0)
            {

                float p = (float)count / nElements;
                entropy -= p * log2(p);
            }
        }
        return entropy;
    }
    float Metrics::informationGain(std::vector<int> &y, std::vector<size_t> &indices, size_t start, size_t end, size_t cutPoint, int nClasses)
    {
        float iGain = 0.0;
        float entropy, entropyLeft, entropyRight;
        int nClassesLeft, nClassesRight;
        int nElementsLeft = cutPoint - start, nElementsRight = end - cutPoint;
        int nElements = end - start;
        nClassesLeft = Metrics::numClasses(y, indices, start, cutPoint);
        nClassesRight = Metrics::numClasses(y, indices, cutPoint, end);
        entropy = Metrics::entropy(y, indices, start, end, nClasses);
        entropyLeft = Metrics::entropy(y, indices, start, cutPoint, nClassesLeft);
        entropyRight = Metrics::entropy(y, indices, cutPoint, end, nClassesRight);
        iGain = entropy - (float)nElementsLeft / nElements * entropyLeft - (float)nElementsRight / nElements * entropyRight;
        return iGain;
    }

}
