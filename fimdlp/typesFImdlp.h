#ifndef TYPES_H
#define TYPES_H
#include <vector>
#include <map>

using namespace std;
namespace mdlp {
    struct CutPointBody {
        size_t start, end;        // indices of the sorted vector
    };
    typedef CutPointBody cutPoint_t;
    typedef vector<float> samples;
    typedef vector<int> labels;
    typedef vector<size_t> indices_t;
    typedef vector<cutPoint_t> cutPoints_t;
    typedef map<tuple<int, int>, float> cacheEnt_t;
    typedef map<tuple<int, int, int>, float> cacheIg_t;
    struct cutPointStruct {
        size_t index;
        float value;
    };
    typedef cutPointStruct xcutPoint_t;
    typedef vector<xcutPoint_t> xcutPoints_t;
}
#endif