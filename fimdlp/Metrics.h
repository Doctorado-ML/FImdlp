#ifndef METRICS_H
#define METRICS_H
#include "typesFImdlp.h"
#include <cmath>
namespace mdlp {
    class Metrics {
    public:
        Metrics();
        static int numClasses(labels&, indices_t, size_t, size_t);
        static float entropy(labels&, indices_t&, size_t, size_t, int);
        static float informationGain(labels&, indices_t&, size_t, size_t, size_t, int);
    };
}
#endif