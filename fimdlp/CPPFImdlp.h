#ifndef CPPFIMDLP_H
#define CPPFIMDLP_H
#include <vector>
#include <utility>
namespace CPPFImdlp
{
    struct CutPointBody
    {
        size_t start, end;
        float fromValue, toValue;
    };
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
        std::vector<CutPointBody> cutPoints(std::vector<float> &, std::vector<int> &);
        std::vector<float> cutPointsAnt(std::vector<float> &, std::vector<int> &);
        void debugPoints(std::vector<float> &, std::vector<int> &);
    };
}
#endif