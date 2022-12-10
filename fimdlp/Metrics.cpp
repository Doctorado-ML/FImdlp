#include "Metrics.h"
#include <set>
#include <cmath>
using namespace std;
namespace mdlp {
    Metrics::Metrics(labels_t& y_, indices_t& indices_): y(y_), indices(indices_), numClasses(computeNumClasses(0, indices.size())), entropyCache(cacheEnt_t()), igCache(cacheIg_t())
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
    void Metrics::setData(labels_t& y_, indices_t& indices_)
    {
        indices = indices_;
        y = y_;
        numClasses = computeNumClasses(0, indices.size());
        entropyCache.clear();
        igCache.clear();
    }
    precision_t Metrics::entropy(size_t start, size_t end)
    {
        precision_t p, ventropy = 0;
        int nElements = 0;
        labels_t counts(numClasses + 1, 0);
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
                p = (precision_t)count / nElements;
                ventropy -= p * log2(p);
            }
        }
        entropyCache[make_tuple(start, end)] = ventropy;
        return ventropy;
    }
    precision_t Metrics::informationGain(size_t start, size_t cut, size_t end)
    {
        precision_t iGain;
        precision_t entropyInterval, entropyLeft, entropyRight;
        int nElementsLeft = cut - start, nElementsRight = end - cut;
        int nElements = end - start;
        if (igCache.find(make_tuple(start, cut, end)) != igCache.end()) {
            return igCache[make_tuple(start, cut, end)];
        }
        entropyInterval = entropy(start, end);
        entropyLeft = entropy(start, cut);
        entropyRight = entropy(cut, end);
        iGain = entropyInterval - ((precision_t)nElementsLeft * entropyLeft + (precision_t)nElementsRight * entropyRight) / nElements;
        igCache[make_tuple(start, cut, end)] = iGain;
        return iGain;
    }

}