// ****************************************************************
// SPDX - FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX - FileType: SOURCE
// SPDX - License - Identifier: MIT
// ****************************************************************

#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <map>
#include <stdexcept>

using namespace std;
namespace mdlp {
    typedef float precision_t;
    typedef int label_t;
    typedef std::vector<precision_t> samples_t;
    typedef std::vector<label_t> labels_t;
    typedef std::vector<size_t> indices_t;
    typedef std::vector<precision_t> cutPoints_t;
    typedef std::map<std::pair<int, int>, precision_t> cacheEnt_t;
    typedef std::map<std::tuple<int, int, int>, precision_t> cacheIg_t;
}
#endif
