// ****************************************************************
// SPDX - FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX - FileType: SOURCE
// SPDX - License - Identifier: MIT
// ****************************************************************

#ifndef CCMETRICS_H
#define CCMETRICS_H

#include "typesFImdlp.h"
#include <mutex>

namespace mdlp {
    class Metrics {
    protected:
        labels_t& y;
        indices_t& indices;
        int numClasses;
        mutable std::mutex cache_mutex;
        cacheEnt_t entropyCache = cacheEnt_t();
        cacheIg_t igCache = cacheIg_t();
    public:
        Metrics(labels_t&, indices_t&);
        void setData(const labels_t&, const indices_t&);
        int computeNumClasses(size_t, size_t);
        precision_t entropy(size_t, size_t);
        precision_t informationGain(size_t, size_t, size_t);
    };
}
#endif