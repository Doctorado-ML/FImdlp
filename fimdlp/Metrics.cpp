#include "Metrics.h"
#include <set>
namespace mdlp {
    Metrics::Metrics()
        = default;
    int Metrics::numClasses(labels& y, indices_t indices, size_t start, size_t end)
    {
        std::set<int> numClasses;
        for (auto i = start; i < end; ++i) {
            numClasses.insert(y[indices[i]]);
        }
        return numClasses.size();
    }
    float Metrics::entropy(labels& y, indices_t& indices, size_t start, size_t end, int nClasses)
    {
        float entropy = 0;
        int nElements = 0;
        labels counts(nClasses + 1, 0);
        for (auto i = &indices[start]; i != &indices[end]; ++i) {
            counts[y[*i]]++;
            nElements++;
        }
        for (auto count : counts) {
            if (count > 0) {
                float p = (float)count / nElements;
                entropy -= p * log2(p);
            }
        }
        return entropy < 0 ? 0 : entropy;
    }
    float Metrics::informationGain(labels& y, indices_t& indices, size_t start, size_t end, size_t cutPoint, int nClasses)
    {
        float iGain;
        float entropy, entropyLeft, entropyRight;
        int nClassesLeft, nClassesRight;
        int nElementsLeft = cutPoint - start, nElementsRight = end - cutPoint;
        int nElements = end - start;
        nClassesLeft = Metrics::numClasses(y, indices, start, cutPoint);
        nClassesRight = Metrics::numClasses(y, indices, cutPoint, end);
        entropy = Metrics::entropy(y, indices, start, end, nClasses);
        entropyLeft = Metrics::entropy(y, indices, start, cutPoint, nClassesLeft);
        entropyRight = Metrics::entropy(y, indices, cutPoint, end, nClassesRight);
        iGain = entropy - ((float)nElementsLeft * entropyLeft + (float)nElementsRight * entropyRight) / nElements;
        return iGain;
    }

}
/*
  cache_t entropyCache;
  std::map<std::tuple<int, int>, double> c;

  // Set the value at index (3, 5) to 7.8.
  c[std::make_tuple(3, 5)] = 7.8;

  // Print the value at index (3, 5).
  std::cout << c[std::make_tuple(3, 5)] << std::endl;
*/