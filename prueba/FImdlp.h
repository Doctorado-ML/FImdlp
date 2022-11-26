#ifndef FIMDLP_H
#define FIMDLP_H
#include <vector>
#include <Python.h>
namespace FImdlp
{
    class FImdlp
    {
    public:
        FImdlp();
        ~FImdlp();
        std::vector<float> cutPoints(std::vector<int> &, std::vector<int> &);
    };
}
#endif