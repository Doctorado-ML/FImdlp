#include "ccMetrics.h"
#include <set>
#include <iostream>
using namespace std;
namespace mdlp {
    Metrics::Metrics(labels& y_, indices_t& indices_): y(y_), indices(indices_), numClasses(computeNumClasses(0, indices.size())), entropyCache(cacheEnt_t()), igCache(cacheIg_t())
    {
    }
    int Metrics::computeNumClasses(size_t start, size_t end)
    {
        set<int> nClasses;
        for (auto i = start; i < end; ++i) {
            nClasses.insert(y[indices[i]]);
        }
        return nClasses.size();
    }
    void Metrics::setData(labels& y_, indices_t& indices_)
    {
        indices = indices_;
        y = y_;
        numClasses = computeNumClasses(0, indices.size());
    }
    float Metrics::entropy(size_t start, size_t end)
    {
        float p, ventropy = 0;
        int nElements = 0;
        labels counts(numClasses + 1, 0);
        if (end - start < 2)
            return 0;
        if (entropyCache.find(make_tuple(start, end)) != entropyCache.end()) {
            return entropyCache[make_tuple(start, end)];
        }
        for (auto i = &indices[start]; i != &indices[end]; ++i) {
            counts[y[*i]]++;
            nElements++;
        }
        for (auto count : counts) {
            if (count > 0) {
                p = (float)count / nElements;
                ventropy -= p * log2(p);
            }
        }
        entropyCache[make_tuple(start, end)] = ventropy;
        return ventropy;
    }
    float Metrics::informationGain(size_t start, size_t cut, size_t end)
    {
        float iGain;
        float entropyInterval, entropyLeft, entropyRight;
        int nElementsLeft = cut - start, nElementsRight = end - cut;
        int nElements = end - start;
        if (igCache.find(make_tuple(start, cut, end)) != igCache.end()) {
            cout << "**********Cache IG hit for " << start << " " << end << endl;
            return igCache[make_tuple(start, cut, end)];
        }
        entropyInterval = entropy(start, end);
        entropyLeft = entropy(start, cut);
        entropyRight = entropy(cut, end);
        iGain = entropyInterval - ((float)nElementsLeft * entropyLeft + (float)nElementsRight * entropyRight) / nElements;
        igCache[make_tuple(start, cut, end)] = iGain;
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