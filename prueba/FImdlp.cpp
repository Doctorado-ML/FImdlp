#include "FImdlp.h"
namespace FImdlp {
    FImdlp::FImdlp()
    {
    }
    FImdlp::~FImdlp()
    {
    }
    samples FImdlp::cutPoints(labels& X, labels& y)
    {
        samples cutPts;
        int i, ant = X.at(0);
        int n = X.size();
        for (i = 1; i < n; i++) {
            if (X.at(i) != ant) {
                cutPts.push_back(precision_t(X.at(i) + ant) / 2);
                ant = X.at(i);
            }
        }
        return cutPts;
    }
}