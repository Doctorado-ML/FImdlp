#ifndef CPPFIMDLP_H
#define CPPFIMDLP_H
#include <vector>
#include <utility>
namespace CPPFImdlp
{
    class CPPFImdlp
    {
    private:
        bool debug;
        int precision;
        float divider;
        std::vector<size_t>
        sortIndices(std::vector<float> &);

    public:
        CPPFImdlp();
        CPPFImdlp(int, bool debug = false);
        ~CPPFImdlp();
        std::vector<float> cutPoints(std::vector<float> &, std::vector<int> &);
        std::vector<float> cutPointsAnt(std::vector<float> &, std::vector<int> &);
    };
}
#endif