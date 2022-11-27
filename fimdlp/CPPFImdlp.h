#ifndef CPPFIMDLP_H
#define CPPFIMDLP_H
#include <vector>
#include <Python.h>
#include <utility>
namespace CPPFImdlp
{
    class CPPFImdlp
    {
    private:
        std::vector<size_t> sortIndices(std::vector<float> &);

    public:
        CPPFImdlp();
        ~CPPFImdlp();
        std::vector<double> cutPoints(std::vector<float> &, std::vector<int> &);
    };
}
#endif