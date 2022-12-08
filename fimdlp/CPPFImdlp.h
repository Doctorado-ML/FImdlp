#ifndef CPPFIMDLP_H
#define CPPFIMDLP_H
#include "typesFImdlp.h"
#include "Metrics.h"
#include <utility>
namespace mdlp {
    class CPPFImdlp {
    protected:
        bool proposal; // proposed algorithm or original algorithm
        bool debug;
        indices_t indices; // sorted indices to use with X and y
        samples X;
        labels y;
        Metrics metrics;
        cutPoints_t cutPoints;

        static indices_t sortIndices(samples&);
        void computeCutPoints(size_t, size_t);
        long int getCandidate(size_t, size_t);
        bool mdlp(size_t, size_t, size_t);

    public:
        CPPFImdlp();
        CPPFImdlp(bool, bool debug = false);
        ~CPPFImdlp();
        CPPFImdlp& fit(samples&, labels&);
        samples getCutPoints();
    };
}
#endif