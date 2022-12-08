#ifndef TYPES_H
#define TYPES_H
#include <vector>
#include <map>

using namespace std;
namespace mdlp {
    typedef float precision_t;
    typedef vector<precision_t> samples;
    typedef vector<int> labels;
    typedef vector<size_t> indices_t;
    typedef vector<precision_t> cutPoints_t;
    typedef map<tuple<int, int>, precision_t> cacheEnt_t;
    typedef map<tuple<int, int, int>, precision_t> cacheIg_t;
}
#endif