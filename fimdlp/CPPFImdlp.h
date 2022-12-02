#ifndef CPPFIMDLP_H
#define CPPFIMDLP_H
#include "typesFImdlp.h"
#include <utility>
namespace mdlp {
    class CPPFImdlp {
    private:
        bool debug;
        int precision;
        float divider;
        indices_t indices; // sorted indices to use with X and y
        samples X;
        labels y;
        labels xDiscretized;
        int numClasses;
        std::vector<CutPoint_t> cutPoints;

    protected:
        indices_t sortIndices(samples&);
        void computeCutPointsAnt();
        void computeCutPoints();
        bool evaluateCutPoint(CutPoint_t, CutPoint_t);
        void filterCutPoints();
        void applyCutPoints();

    public:
        CPPFImdlp();
        CPPFImdlp(int, bool debug = false);
        ~CPPFImdlp();
        std::vector<CutPoint_t> getCutPoints();
        labels getDiscretizedValues();
        void debugPoints(samples&, labels&);
        void fit(samples&, labels&);
        labels& transform(samples&);
    };
}
#endif