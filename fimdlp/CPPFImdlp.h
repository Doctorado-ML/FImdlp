#ifndef CPPFIMDLP_H
#define CPPFIMDLP_H
#include "typesFImdlp.h"
#include "Metrics.h"
#include <utility>
namespace mdlp {
    class CPPFImdlp {
    protected:
        bool proposal;
        indices_t indices; // sorted indices to use with X and y
        samples_t X;
        labels_t y;
        Metrics metrics;
        cutPoints_t cutPoints;

        static indices_t sortIndices(samples_t&);
        void computeCutPoints(size_t, size_t);
        long int getCandidate(size_t, size_t);
        bool mdlp(size_t, size_t, size_t);

        // Original algorithm
        void computeCutPointsOriginal(size_t, size_t);
        bool goodCut(size_t, size_t, size_t);
        void computeCutPointsProposal();

    public:
        CPPFImdlp(bool);
        ~CPPFImdlp();
        CPPFImdlp& fit(samples_t&, labels_t&);
        samples_t getCutPoints();
    };
}
#endif