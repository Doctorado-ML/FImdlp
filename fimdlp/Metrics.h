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
        static float entropy(std::vector<int> &, int, int, int);
        static int numClasses(std::vector<int> &);
    };
}
#endif