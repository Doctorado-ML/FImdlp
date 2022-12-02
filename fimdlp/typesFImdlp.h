#ifndef TYPES_H
#define TYPES_H
#include <vector>
namespace mdlp {
    struct CutPointBody {
        size_t start, end;        // indices of the sorted vector
        int classNumber;          // class assigned to the cut point
        float fromValue, toValue;
    };
    typedef CutPointBody cutPoint_t;
    typedef std::vector<float> samples;
    typedef std::vector<int> labels;
    typedef std::vector<size_t> indices_t;
    typedef std::vector<cutPoint_t> cutPoints_t;
}
#endif