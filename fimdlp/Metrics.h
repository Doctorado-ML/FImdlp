#ifndef CCMETRICS_H
#define CCMETRICS_H
#include "typesFImdlp.h"
#include <cmath>
namespace mdlp {
    class Metrics {
    protected:
        labels& y;
        indices_t& indices;
        int numClasses;
        cacheEnt_t entropyCache;
        cacheIg_t igCache;
    public:
        Metrics(labels&, indices_t&);
        void setData(labels&, indices_t&);
        int computeNumClasses(size_t, size_t);
        precision_t entropy(size_t, size_t);
        precision_t informationGain(size_t, size_t, size_t);
    };
}
#endif