#ifndef CPPFIMDLP_H
#define CPPFIMDLP_H
#include <vector>
#include <Python.h>
namespace CPPFImdlp
{
    class CPPFImdlp
    {
    public:
        CPPFImdlp();
        ~CPPFImdlp();
        std::vector<float> cutPoints(std::vector<int> &, std::vector<int> &);
    };
}
#endif