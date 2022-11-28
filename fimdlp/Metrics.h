#ifndef METRICS_H
#define METRICS_H
#include <vector>
#include <Python.h>
#include <utility>
namespace CPPFImdlp
{
    class Metrics
    {
    public:
        Metrics();
        static int numClasses(std::vector<int> &, std::vector<size_t>, int, int);
        static float entropy(std::vector<int> &, std::vector<size_t> &, int, int, int);
        static float informationGain(std::vector<int> &y, std::vector<size_t> &indices, int start, int end, int cutPoint, int nClasses);
    };
}
#endif