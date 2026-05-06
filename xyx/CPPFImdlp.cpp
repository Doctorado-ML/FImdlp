// ****************************************************************
// SPDX - FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX - FileType: SOURCE
// SPDX - License - Identifier: MIT
// ****************************************************************

#include <numeric>
#include <algorithm>
#include <set>
#include <cmath>
#include <stdexcept>
#include "CPPFImdlp.h"

namespace mdlp {

    CPPFImdlp::CPPFImdlp(size_t min_length_, int max_depth_, float proposed) :
        Discretizer(),
        min_length(min_length_),
        max_depth(max_depth_),
        proposed_cuts(proposed)
    {
        // Input validation for constructor parameters
        if (min_length_ < 3) {
            throw std::invalid_argument("min_length must be greater than 2");
        }
        if (max_depth_ < 1) {
            throw std::invalid_argument("max_depth must be greater than 0");
        }
        if (proposed < 0.0f) {
            throw std::invalid_argument("proposed_cuts must be non-negative");
        }

        direction = bound_dir_t::RIGHT;
    }

    size_t CPPFImdlp::compute_max_num_cut_points() const
    {
        // Set the actual maximum number of cut points as a number or as a percentage of the number of samples
        if (proposed_cuts == 0) {
            return numeric_limits<size_t>::max();
        }
        if (proposed_cuts > static_cast<precision_t>(X.size())) {
            throw invalid_argument("wrong proposed num_cuts value");
        }
        if (proposed_cuts < 1)
            return static_cast<size_t>(round(static_cast<precision_t>(X.size()) * proposed_cuts));
        return static_cast<size_t>(proposed_cuts); // The 2 extra cutpoints should not be considered here as this parameter is considered before they are added
    }

    void CPPFImdlp::fit(samples_t& X_, labels_t& y_)
    {
        X = X_;
        y = y_;
        num_cut_points = compute_max_num_cut_points();
        depth = 0;
        discretizedData.clear();
        cutPoints.clear();
        if (X.size() != y.size()) {
            throw std::invalid_argument("X and y must have the same size: " + std::to_string(X.size()) + " != " + std::to_string(y.size()));
        }
        if (X.empty() || y.empty()) {
            throw invalid_argument("X and y must have at least one element");
        }
        indices = sortIndices(X_, y_);
        metrics.setData(y, indices);
        computeCutPoints(0, X.size(), 1);
        sort(cutPoints.begin(), cutPoints.end());
        if (num_cut_points > 0) {
            // Select the best (with lower entropy) cut points
            while (cutPoints.size() > num_cut_points) {
                resizeCutPoints();
            }
        }
        // Insert first & last X value to the cutpoints as them shall be ignored in transform
        auto [vmin, vmax] = std::minmax_element(X.begin(), X.end());
        cutPoints.push_back(*vmax);
        cutPoints.insert(cutPoints.begin(), *vmin);
    }

    pair<precision_t, size_t> CPPFImdlp::valueCutPoint(size_t start, size_t cut, size_t end)
    {
        size_t n;
        size_t m;
        size_t idxPrev = cut - 1 >= start ? cut - 1 : cut;
        size_t idxNext = cut + 1 < end ? cut + 1 : cut;
        bool backWall; // true if duplicates reach beginning of the interval
        precision_t previous;
        precision_t actual;
        precision_t next;
        previous = safe_X_access(idxPrev);
        actual = safe_X_access(cut);
        next = safe_X_access(idxNext);
        // definition 2 of the paper => X[t-1] < X[t]
        // get the first equal value of X in the interval
        while (idxPrev > start && actual == previous) {
            --idxPrev;
            previous = safe_X_access(idxPrev);
        }
        backWall = idxPrev == start && actual == previous;
        // get the last equal value of X in the interval
        while (idxNext < end - 1 && actual == next) {
            ++idxNext;
            next = safe_X_access(idxNext);
        }
        // # of duplicates before cutpoint
        n = safe_subtract(safe_subtract(cut, 1), idxPrev);
        // # of duplicates after cutpoint
        m = idxNext - cut - 1;
        // Decide which values to use
        if (backWall) {
            m = int(idxNext - cut - 1) < 0 ? 0 : m; // Ensure m right
            cut = cut + m + 1;
        } else {
            cut = safe_subtract(cut, n);
        }
        actual = safe_X_access(cut);
        return { (actual + previous) / 2, cut };
    }

    void CPPFImdlp::computeCutPoints(size_t start, size_t end, int depth_)
    {
        size_t cut;
        pair<precision_t, size_t> result;
        // Check if the interval length and the depth are Ok
        if (end < start || safe_subtract(end, start) < min_length || depth_ > max_depth)
            return;
        depth = depth_ > depth ? depth_ : depth;
        cut = getCandidate(start, end);
        if (cut == numeric_limits<size_t>::max())
            return;
        if (mdlp(start, cut, end)) {
            result = valueCutPoint(start, cut, end);
            cut = result.second;
            cutPoints.push_back(result.first);
            computeCutPoints(start, cut, depth_ + 1);
            computeCutPoints(cut, end, depth_ + 1);
        }
    }

    size_t CPPFImdlp::getCandidate(size_t start, size_t end)
    {
        /* Definition 1: A binary discretization for A is determined by selecting the cut point TA for which
        E(A, TA; S) is minimal amongst all the candidate cut points. */
        size_t candidate = numeric_limits<size_t>::max();
        size_t elements = safe_subtract(end, start);
        bool sameValues = true;
        precision_t entropy_left;
        precision_t entropy_right;
        precision_t minEntropy;
        // Check if all the values of the variable in the interval are the same
        for (size_t idx = start + 1; idx < end; idx++) {
            if (safe_X_access(idx) != safe_X_access(start)) {
                sameValues = false;
                break;
            }
        }
        if (sameValues)
            return candidate;
        minEntropy = metrics.entropy(start, end);
        for (size_t idx = start + 1; idx < end; idx++) {
            // Cutpoints are always on boundaries (definition 2)
            if (safe_y_access(idx) == safe_y_access(idx - 1))
                continue;
            entropy_left = precision_t(idx - start) / static_cast<precision_t>(elements) * metrics.entropy(start, idx);
            entropy_right = precision_t(end - idx) / static_cast<precision_t>(elements) * metrics.entropy(idx, end);
            if (entropy_left + entropy_right < minEntropy) {
                minEntropy = entropy_left + entropy_right;
                candidate = idx;
            }
        }
        return candidate;
    }

    bool CPPFImdlp::mdlp(size_t start, size_t cut, size_t end)
    {
        int k;
        int k1;
        int k2;
        precision_t ig;
        precision_t delta;
        precision_t ent;
        precision_t ent1;
        precision_t ent2;
        auto N = precision_t(safe_subtract(end, start));
        k = metrics.computeNumClasses(start, end);
        k1 = metrics.computeNumClasses(start, cut);
        k2 = metrics.computeNumClasses(cut, end);
        ent = metrics.entropy(start, end);
        ent1 = metrics.entropy(start, cut);
        ent2 = metrics.entropy(cut, end);
        ig = metrics.informationGain(start, cut, end);
        delta = static_cast<precision_t>(log2(pow(3, precision_t(k)) - 2) -
            (precision_t(k) * ent - precision_t(k1) * ent1 - precision_t(k2) * ent2));
        precision_t term = 1 / N * (log2(N - 1) + delta);
        return ig > term;
    }

    // Argsort from https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
    indices_t CPPFImdlp::sortIndices(samples_t& X_, labels_t& y_)
    {
        indices_t idx(X_.size());
        std::iota(idx.begin(), idx.end(), 0);
        stable_sort(idx.begin(), idx.end(), [&X_, &y_](size_t i1, size_t i2) {
            if (i1 >= X_.size() || i2 >= X_.size() || i1 >= y_.size() || i2 >= y_.size()) {
                throw std::out_of_range("Index out of bounds in sort comparison");
            }
            if (X_[i1] == X_[i2])
                return y_[i1] < y_[i2];
            else
                return X_[i1] < X_[i2];
            });
        return idx;
    }

    void CPPFImdlp::resizeCutPoints()
    {
        //Compute entropy of each of the whole cutpoint set and discards the biggest value
        precision_t maxEntropy = 0;
        precision_t entropy;
        size_t maxEntropyIdx = 0;
        size_t begin = 0;
        size_t end;
        for (size_t idx = 0; idx < cutPoints.size(); idx++) {
            end = begin;
            while (end < indices.size() && safe_X_access(end) < cutPoints[idx] && end < X.size())
                end++;
            entropy = metrics.entropy(begin, end);
            if (entropy > maxEntropy) {
                maxEntropy = entropy;
                maxEntropyIdx = idx;
            }
            begin = end;
        }
        cutPoints.erase(cutPoints.begin() + static_cast<long>(maxEntropyIdx));
    }

}
