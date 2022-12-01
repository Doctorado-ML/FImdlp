#include "gtest/gtest.h"
#include "../CPPFImdlp.h"
namespace
{
    float precision = 0.000001;
    class TestMetrics : protected mdlp::CPPFImdlp
    {
    public:
        std::vector<size_t> testSort(std::vector<float> &X)
        {
            return sortIndices(X);
        }
    };
    void check_sorted_vector(std::vector<float> &X, std::vector<size_t> indices)
    {
        TestMetrics testClass = TestMetrics();
        std::vector<size_t> testSortedIndices = testClass.testSort(X);
        float prev = X[testSortedIndices[0]];
        for (auto i = 0; i < X.size(); ++i)
        {
            EXPECT_EQ(testSortedIndices[i], indices[i]);
            EXPECT_LE(prev, X[testSortedIndices[i]]);
            prev = X[testSortedIndices[i]];
        }
    }
    TEST(FImdlpTest, SortIndices)
    {

        std::vector<float> X = {5.7, 5.3, 5.2, 5.1, 5.0, 5.6, 5.1, 6.0, 5.1, 5.9};
        std::vector<size_t> indices = {4, 3, 6, 8, 2, 1, 5, 0, 9, 7};
        check_sorted_vector(X, indices);
        X = {5.77, 5.88, 5.99};
        indices = {0, 1, 2};
        check_sorted_vector(X, indices);
        X = {5.33, 5.22, 5.11};
        indices = {2, 1, 0};
        check_sorted_vector(X, indices);
    }
}