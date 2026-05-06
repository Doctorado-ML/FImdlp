// ****************************************************************
// SPDX - FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX - FileType: SOURCE
// SPDX - License - Identifier: MIT
// ****************************************************************

#ifndef BINDISC_H
#define BINDISC_H

#include "typesFImdlp.h"
#include "Discretizer.h"
#include <string>

namespace mdlp {
    enum class strategy_t {
        UNIFORM, // Equal width
        QUANTILE // Equal frequency
    };
    class BinDisc : public Discretizer {
    public:
        BinDisc(int n_bins = 3, strategy_t strategy = strategy_t::UNIFORM);
        ~BinDisc();
        // y is included for compatibility with the Discretizer interface
        void fit(samples_t& X_, labels_t& y) override;
        void fit(samples_t& X);
    protected:
        std::vector<precision_t> linspace(precision_t start, precision_t end, int num);
        std::vector<precision_t> percentile(samples_t& data, const std::vector<precision_t>& percentiles);
        int n_bins;
        strategy_t strategy;
        const int min_bins = 3;
    private:
        void fit_uniform(const samples_t&);
        void fit_quantile(const samples_t&);
    };
}
#endif
