#ifndef FIMDLP_H
#define FIMDLP_H
#include <vector>
#include <Python.h>
namespace FImdlp {
    class FImdlp {
    public:
        FImdlp();
        ~FImdlp();
        samples cutPoints(labels&, labels&);
    };
}
#endif