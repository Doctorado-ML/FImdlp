
#include <vector>

using namespace std;
struct CutPointBody {
    size_t start, end;        // indices of the sorted vector
    int classNumber;          // class assigned to the cut point
    float fromValue, toValue;
};
typedef CutPointBody cutPoint_t;
typedef vector<float> samples;
typedef vector<int> labels;
typedef vector<size_t> indices_t;
typedef vector<cutPoint_t> cutPoints_t;
//typedef std::map<std::tuple<int, int>, float> cache_t;
struct cutPointStruct {
    size_t index;
    float value;
};
typedef cutPointStruct xcutPoint_t;
typedef vector<xcutPoint_t> xcutPoints_t;
class Metrics {
private:
    labels& y;
    indices_t& indices;
    int numClasses;
public:
    Metrics(labels&, indices_t&);
    int computeNumClasses(size_t, size_t);
    float entropy(size_t, size_t);
    float informationGain(size_t, size_t, size_t);
};
Metrics::Metrics(labels& y_, indices_t& indices_) : y(y_), indices(indices_)
{
    numClasses = computeNumClasses(0, indices.size());
}