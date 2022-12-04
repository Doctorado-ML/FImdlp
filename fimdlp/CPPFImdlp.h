#ifndef CPPFIMDLP_H
#define CPPFIMDLP_H
#include "typesFImdlp.h"
#include <utility>
namespace mdlp {
    class CPPFImdlp {
    protected:
        bool proposed; // proposed algorithm or original algorithm
        int precision;
        bool debug;
        float divider;
        indices_t indices; // sorted indices to use with X and y
        samples X;
        labels y;
        labels xDiscretized;
        int numClasses;
        cutPoints_t cutPoints;

        void setCutPoints(cutPoints_t);
        static indices_t sortIndices(samples&);
        void computeCutPointsOriginal();
        void computeCutPointsProposed();
        bool evaluateCutPoint(cutPoint_t, cutPoint_t);
        void filterCutPoints();
        void applyCutPoints();

    public:
        CPPFImdlp();
        CPPFImdlp(bool, int, bool debug = false);
        ~CPPFImdlp();
        cutPoints_t getCutPoints();
        indices_t getIndices();
        labels getDiscretizedValues();
        void debugPoints(samples&, labels&);
        CPPFImdlp& fit(samples&, labels&);
        labels& transform(samples&);
    };
}
#endif