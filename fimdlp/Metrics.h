#ifndef METRICS_H
#define METRICS_H
#include <vector>
#include <cmath>
namespace CPPFImdlp
{
    class Metrics
    {
    public:
        Metrics();
        static int numClasses(std::vector<int> &, std::vector<size_t>, size_t, size_t);
        static float entropy(std::vector<int> &, std::vector<size_t> &, size_t, size_t, int);
        static float informationGain(std::vector<int> &, std::vector<size_t> &, size_t, size_t, size_t, int);
    };
}
#endif