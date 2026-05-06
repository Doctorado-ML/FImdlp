// ****************************************************************
// SPDX - FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX - FileType: SOURCE
// SPDX - License - Identifier: MIT
// ****************************************************************

#include "Discretizer.h"

namespace mdlp {

    labels_t& Discretizer::transform(const samples_t& data)
    {
        // Input validation
        if (data.empty()) {
            throw std::invalid_argument("Data for transformation cannot be empty");
        }
        if (cutPoints.size() < 2) {
            throw std::runtime_error("Discretizer not fitted yet or no valid cut points found");
        }

        discretizedData.clear();
        discretizedData.reserve(data.size());
        // CutPoints always have at least two items
        // Have to ignore first and last cut points provided
        auto first = cutPoints.begin() + 1;
        auto last = cutPoints.end() - 1;
        auto bound = direction == bound_dir_t::LEFT ? std::lower_bound<std::vector<precision_t>::iterator, precision_t> : std::upper_bound<std::vector<precision_t>::iterator, precision_t>;
        for (const precision_t& item : data) {
            auto pos = bound(first, last, item);
            auto number = pos - first;
            discretizedData.push_back(static_cast<label_t>(number));
        }
        return discretizedData;
    }
    labels_t& Discretizer::fit_transform(samples_t& X_, labels_t& y_)
    {
        fit(X_, y_);
        return transform(X_);
    }
    void Discretizer::fit_t(const torch::Tensor& X_, const torch::Tensor& y_)
    {
        // Validate tensor properties for security
        if (X_.sizes().size() != 1 || y_.sizes().size() != 1) {
            throw std::invalid_argument("Only 1D tensors supported");
        }
        if (X_.dtype() != torch::kFloat32) {
            throw std::invalid_argument("X tensor must be Float32 type");
        }
        if (y_.dtype() != torch::kInt32) {
            throw std::invalid_argument("y tensor must be Int32 type");
        }
        if (X_.numel() != y_.numel()) {
            throw std::invalid_argument("X and y tensors must have same number of elements");
        }
        if (X_.numel() == 0) {
            throw std::invalid_argument("Tensors cannot be empty");
        }

        auto num_elements = X_.numel();
        samples_t X(X_.data_ptr<precision_t>(), X_.data_ptr<precision_t>() + num_elements);
        labels_t y(y_.data_ptr<int>(), y_.data_ptr<int>() + num_elements);
        fit(X, y);
    }
    torch::Tensor Discretizer::transform_t(const torch::Tensor& X_)
    {
        // Validate tensor properties for security
        if (X_.sizes().size() != 1) {
            throw std::invalid_argument("Only 1D tensors supported");
        }
        if (X_.dtype() != torch::kFloat32) {
            throw std::invalid_argument("X tensor must be Float32 type");
        }
        if (X_.numel() == 0) {
            throw std::invalid_argument("Tensor cannot be empty");
        }

        auto num_elements = X_.numel();
        samples_t X(X_.data_ptr<precision_t>(), X_.data_ptr<precision_t>() + num_elements);
        auto result = transform(X);
        return torch::tensor(result, torch_label_t);
    }
    torch::Tensor Discretizer::fit_transform_t(const torch::Tensor& X_, const torch::Tensor& y_)
    {
        // Validate tensor properties for security
        if (X_.sizes().size() != 1 || y_.sizes().size() != 1) {
            throw std::invalid_argument("Only 1D tensors supported");
        }
        if (X_.dtype() != torch::kFloat32) {
            throw std::invalid_argument("X tensor must be Float32 type");
        }
        if (y_.dtype() != torch::kInt32) {
            throw std::invalid_argument("y tensor must be Int32 type");
        }
        if (X_.numel() != y_.numel()) {
            throw std::invalid_argument("X and y tensors must have same number of elements");
        }
        if (X_.numel() == 0) {
            throw std::invalid_argument("Tensors cannot be empty");
        }

        auto num_elements = X_.numel();
        samples_t X(X_.data_ptr<precision_t>(), X_.data_ptr<precision_t>() + num_elements);
        labels_t y(y_.data_ptr<int>(), y_.data_ptr<int>() + num_elements);
        auto result = fit_transform(X, y);
        return torch::tensor(result, torch_label_t);
    }
}