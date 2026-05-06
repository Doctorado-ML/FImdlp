// ****************************************************************
// SPDX - FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX - FileType: SOURCE
// SPDX - License - Identifier: MIT
// ****************************************************************

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include "BinDisc.h"

namespace mdlp {

    BinDisc::BinDisc(int n_bins, strategy_t strategy) :
        Discretizer(), n_bins{ n_bins }, strategy{ strategy }
    {
        if (n_bins < min_bins) {
            throw std::invalid_argument("n_bins must be greater than " + std::to_string(min_bins - 1));
        }
    }
    BinDisc::~BinDisc() = default;
    void BinDisc::fit(samples_t& X)
    {
        // Input validation
        if (X.empty()) {
            throw std::invalid_argument("Input data X cannot be empty");
        }
        if (X.size() < static_cast<size_t>(n_bins)) {
            throw std::invalid_argument("Input data size must be at least equal to n_bins");
        }

        cutPoints.clear();
        if (strategy == strategy_t::QUANTILE) {
            direction = bound_dir_t::RIGHT;
            fit_quantile(X);
        } else if (strategy == strategy_t::UNIFORM) {
            direction = bound_dir_t::RIGHT;
            fit_uniform(X);
        }
    }
    void BinDisc::fit(samples_t& X, labels_t& y)
    {
        if (X.empty()) {
            throw std::invalid_argument("X cannot be empty");
        }

        // BinDisc is inherently unsupervised, but we validate inputs for consistency
        // Note: y parameter is validated but not used in binning strategy
        fit(X);
    }
    std::vector<precision_t> BinDisc::linspace(precision_t start, precision_t end, int num)
    {
        // Input validation
        if (num < 2) {
            throw std::invalid_argument("Number of points must be at least 2 for linspace");
        }
        if (std::isnan(start) || std::isnan(end)) {
            throw std::invalid_argument("Start and end values cannot be NaN");
        }
        if (std::isinf(start) || std::isinf(end)) {
            throw std::invalid_argument("Start and end values cannot be infinite");
        }

        if (start == end) {
            return { start, end };
        }
        precision_t delta = (end - start) / static_cast<precision_t>(num - 1);
        std::vector<precision_t> linspc;
        for (size_t i = 0; i < num; ++i) {
            precision_t val = start + delta * static_cast<precision_t>(i);
            linspc.push_back(val);
        }
        return linspc;
    }
    size_t clip(const size_t n, const size_t lower, const size_t upper)
    {
        return std::max(lower, std::min(n, upper));
    }
    std::vector<precision_t> BinDisc::percentile(samples_t& data, const std::vector<precision_t>& percentiles)
    {
        // Input validation
        if (data.empty()) {
            throw std::invalid_argument("Data cannot be empty for percentile calculation");
        }
        if (percentiles.empty()) {
            throw std::invalid_argument("Percentiles cannot be empty");
        }

        // Implementation taken from https://dpilger26.github.io/NumCpp/doxygen/html/percentile_8hpp_source.html
        std::vector<precision_t> results;
        bool first = true;
        results.reserve(percentiles.size());
        for (auto percentile : percentiles) {
            const auto i = static_cast<size_t>(std::floor(static_cast<precision_t>(data.size() - 1) * percentile / 100.));
            const auto indexLower = clip(i, 0, data.size() - 2);
            const precision_t percentI = static_cast<precision_t>(indexLower) / static_cast<precision_t>(data.size() - 1);
            const precision_t fraction =
                (percentile / 100.0 - percentI) /
                (static_cast<precision_t>(indexLower + 1) / static_cast<precision_t>(data.size() - 1) - percentI);
            if (const auto value = data[indexLower] + (data[indexLower + 1] - data[indexLower]) * fraction; first || results.empty() || value != results.back()) // Check empty before calling back()
                results.push_back(value);
            first = false;
        }
        return results;
    }
    void BinDisc::fit_quantile(const samples_t& X)
    {
        auto quantiles = linspace(0.0, 100.0, n_bins + 1);
        auto data = X;
        std::sort(data.begin(), data.end());
        if (data.front() == data.back() || data.size() == 1) {
            // if X is constant, pass any two given points that shall be ignored in transform
            cutPoints.push_back(data.front());
            cutPoints.push_back(data.front());
            return;
        }
        cutPoints = percentile(data, quantiles);
    }
    void BinDisc::fit_uniform(const samples_t& X)
    {
        auto [vmin, vmax] = std::minmax_element(X.begin(), X.end());
        cutPoints = linspace(*vmin, *vmax, n_bins + 1);
    }
}