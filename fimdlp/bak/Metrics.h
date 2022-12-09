#ifndef METRICS_H
#define METRICS_H
#include "typesFImdlp.h"
#include <cmath>
namespace mdlp {
    class Metrics {
    public:
        Metrics();
        static int numClasses(labels&, indices_t, size_t, size_t);
        static precision_t entropy(labels&, indices_t&, size_t, size_t, int);
        static precision_t informationGain(labels&, indices_t&, size_t, size_t, size_t, int);
    };
}
#endif