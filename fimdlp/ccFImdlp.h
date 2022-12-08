#ifndef CCFIMDLP_H
#define CCFIMDLP_H
#include "typesFImdlp.h"
#include "ccMetrics.h"
#include <utility>
namespace mdlp {
    class CPPFImdlp {
    protected:
        bool proposal; // proposed algorithm or original algorithm
        int precision;
        bool debug;
        float divider;
        indices_t indices; // sorted indices to use with X and y
        samples X;
        labels y;
        Metrics metrics;
        xcutPoints_t xCutPoints;

        static indices_t sortIndices(samples&);
        void computeCutPoints(size_t, size_t);
        long int getCandidate(size_t, size_t);
        bool mdlp(size_t, size_t, size_t);

    public:
        CPPFImdlp();
        CPPFImdlp(bool, int, bool debug = false);
        ~CPPFImdlp();
        CPPFImdlp& fitx(samples&, labels&);
        samples getCutPointsx();
    };
}
#endif