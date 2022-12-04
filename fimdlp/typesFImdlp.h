#ifndef TYPES_H
#define TYPES_H
#include <vector>

using namespace std;
namespace mdlp {
    struct CutPointBody {
        size_t start, end;        // indices of the sorted vector
        int classNumber;          // class assigned to the cut point
        float fromValue, toValue;
    };
    typedef CutPointBody cutPoint_t;
    typedef vector<float> samples;
    typedef vector<int> labels;
    typedef vector<size_t> indices_t;
    typedef vector<cutPoint_t> cutPoints_t;
}
#endif