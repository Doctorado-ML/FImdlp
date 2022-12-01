#ifndef CPPFIMDLP_H
#define CPPFIMDLP_H
#include <vector>
#include <utility>
namespace mdlp
{
    struct CutPointBody
    {
        size_t start, end;        // indices of the sorted vector
        int classNumber;          // class assigned to the cut point
        float fromValue, toValue; // Values of the variable
    };
    class CPPFImdlp
    {
    private:
        bool debug;
        int precision;
        float divider;
        std::vector<size_t> indices; // sorted indices to use with X and y
        std::vector<float> X;
        std::vector<int> y;
        std::vector<float> xDiscretized;
        std::vector<CutPointBody> cutPoints;

    protected:
        std::vector<size_t> sortIndices(std::vector<float> &);

    public:
        CPPFImdlp();
        CPPFImdlp(int, bool debug = false);
        ~CPPFImdlp();
        std::vector<CutPointBody> getCutPoints();
        void computeCutPoints(std::vector<float> &, std::vector<int> &);
        std::vector<float> computeCutPointsAnt(std::vector<float> &, std::vector<int> &);
        void debugPoints(std::vector<float> &, std::vector<int> &);
    };
}
#endif