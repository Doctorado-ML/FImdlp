#ifndef CPPFIMDLP_H
#define CPPFIMDLP_H
#include "typesFImdlp.h"
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
        labels xDiscretized;
        int numClasses;
        cutPoints_t cutPoints;

        void setCutPoints(cutPoints_t);
        static indices_t sortIndices(samples&);
        void computeCutPointsOriginal();
        void computeCutPointsProposal();
        bool evaluateCutPoint(cutPoint_t, cutPoint_t);
        void filterCutPoints();
        bool goodCut(size_t, size_t, size_t); // if the cut candidate reduces entropy

    public:
        CPPFImdlp();
        CPPFImdlp(bool, int, bool debug = false);
        ~CPPFImdlp();
        samples getCutPoints();
        indices_t getIndices();
        labels getDiscretizedValues();
        void debugPoints(samples&, labels&);
        CPPFImdlp& fit(samples&, labels&);
        labels transform(samples&);
    };
}
#endif