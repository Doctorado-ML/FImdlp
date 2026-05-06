// ****************************************************************
// SPDX - FileCopyrightText: Copyright 2025 Ricardo Montañana Gómez
// SPDX - FileType: SOURCE
// SPDX - License - Identifier: MIT
// ****************************************************************

#include "PKIDisc.h"

namespace mdlp {

    void PKIDisc::fit(samples_t& X, labels_t& y)
    {
        if (compute_strategy == compute_strategy_t::LOG) {
            n_bins = static_cast<int>(std::log(static_cast<int>(y.size())));
        } else {
            n_bins = static_cast<int>(sqrt(static_cast<int>(y.size())));
        }
        strategy = strategy_t::QUANTILE;
        if (n_bins < min_bins) {
            n_bins = min_bins;
        }
        BinDisc::fit(X, y);
    }
}
