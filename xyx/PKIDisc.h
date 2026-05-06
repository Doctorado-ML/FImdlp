// ****************************************************************
// SPDX - FileCopyrightText: Copyright 2025 Ricardo Montañana Gómez
// SPDX - FileType: SOURCE
// SPDX - License - Identifier: MIT
// ****************************************************************

#ifndef PKIDISC_H
#define PKIDISC_H

#include "BinDisc.h"

namespace mdlp {
    enum class compute_strategy_t {
        LOG, // Logarithmic
        SQRT // Square root
    };
    class PKIDisc : public BinDisc {
    public:
        PKIDisc(compute_strategy_t compute_strategy_ = compute_strategy_t::SQRT) : compute_strategy(compute_strategy_) {}
        ~PKIDisc() = default;
        void fit(samples_t& X_, labels_t& y) override;
    private:
        compute_strategy_t compute_strategy;
    };
}
#endif
