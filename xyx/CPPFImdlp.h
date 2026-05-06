// ****************************************************************
// SPDX - FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX - FileType: SOURCE
// SPDX - License - Identifier: MIT
// ****************************************************************

#ifndef CPPFIMDLP_H
#define CPPFIMDLP_H

#include "typesFImdlp.h"
#include <limits>
#include <utility>
#include <string>
#include "Metrics.h"
#include "Discretizer.h"

namespace mdlp {
    class CPPFImdlp : public Discretizer {
    public:
        CPPFImdlp() = default;
        CPPFImdlp(size_t min_length_, int max_depth_, float proposed);
        virtual ~CPPFImdlp() = default;
        void fit(samples_t& X_, labels_t& y_) override;
        inline int get_depth() const { return depth; };
    protected:
        size_t min_length = 3;
        int depth = 0;
        int max_depth = numeric_limits<int>::max();
        float proposed_cuts = 0;
        indices_t indices = indices_t();
        samples_t X = samples_t();
        labels_t y = labels_t();
        Metrics metrics = Metrics(y, indices);
        size_t num_cut_points = numeric_limits<size_t>::max();
        static indices_t sortIndices(samples_t&, labels_t&);
        void computeCutPoints(size_t, size_t, int);
        void resizeCutPoints();
        bool mdlp(size_t, size_t, size_t);
        size_t getCandidate(size_t, size_t);
        size_t compute_max_num_cut_points() const;
        pair<precision_t, size_t> valueCutPoint(size_t, size_t, size_t);
        inline precision_t safe_X_access(size_t idx) const
        {
            if (indices.empty()) {
                throw std::out_of_range("Indices array is empty");
            }
            if (idx >= indices.size()) {
                throw std::out_of_range("Index out of bounds for indices array");
            }
            size_t real_idx = indices[idx];
            if (real_idx >= X.size()) {
                throw std::out_of_range("Index out of bounds for X array");
            }
            return X[real_idx];
        }
        inline label_t safe_y_access(size_t idx) const
        {
            if (indices.empty()) {
                throw std::out_of_range("Indices array is empty");
            }
            if (idx >= indices.size()) {
                throw std::out_of_range("Index out of bounds for indices array");
            }
            size_t real_idx = indices[idx];
            if (real_idx >= y.size()) {
                throw std::out_of_range("Index out of bounds for y array");
            }
            return y[real_idx];
        }
        inline size_t safe_subtract(size_t a, size_t b) const
        {
            if (b > a) {
                throw std::underflow_error("Subtraction would cause underflow");
            }
            return a - b;
        }
    };
}
#endif
