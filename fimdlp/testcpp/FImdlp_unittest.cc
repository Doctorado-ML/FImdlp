#include "gtest/gtest.h"
#include "../Metrics.h"
#include "../CPPFImdlp.h"
namespace mdlp {
    class TestMetrics : public CPPFImdlp, public testing::Test {
    public:
        //TestMetrics(samples X, labels y, indices_t indices) : X(X), y(y), indices(indices), CPPFImdlp(true) {}
        indices_t indices; // sorted indices to use with X and y
        samples X;
        labels y;
        samples xDiscretized;
        int numClasses;
        float precision_test = 0.000001;
        void SetUp() override
        {
            X = { 5.7, 5.3, 5.2, 5.1, 5.0, 5.6, 5.1, 6.0, 5.1, 5.9 };
            indices = { 4, 3, 6, 8, 2, 1, 5, 0, 9, 7 };
            y = { 1, 1, 1, 1, 1, 2, 2, 2, 2, 2 };
            numClasses = 2;
        }
        void check_sorted_vector(samples& X, indices_t indices)
        {
            this->X = X;
            this->indices = indices;
            indices_t testSortedIndices = sortIndices(X);
            float prev = X[testSortedIndices[0]];
            for (auto i = 0; i < X.size(); ++i) {
                EXPECT_EQ(testSortedIndices[i], indices[i]);
                EXPECT_LE(prev, X[testSortedIndices[i]]);
                prev = X[testSortedIndices[i]];
            }
        }
        std::vector<CutPoint_t> testCutPoints(samples& X, indices_t& indices, labels& y)
        {
            this->X = X;
            this->y = y;
            this->indices = indices;
            this->numClasses = Metrics::numClasses(y, indices, 0, X.size());

            //computeCutPoints();
            return getCutPoints();
        }
    };
    // 
    TEST_F(TestMetrics, SortIndices)
    {
        samples X = { 5.7, 5.3, 5.2, 5.1, 5.0, 5.6, 5.1, 6.0, 5.1, 5.9 };
        indices_t indices = { 4, 3, 6, 8, 2, 1, 5, 0, 9, 7 };
        check_sorted_vector(X, indices);
        X = { 5.77, 5.88, 5.99 };
        indices = { 0, 1, 2 };
        check_sorted_vector(X, indices);
        X = { 5.33, 5.22, 5.11 };
        indices = { 2, 1, 0 };
        check_sorted_vector(X, indices);
    }
    // TEST_F(TestMetrics, EvaluateCutPoint)
    // {
    //     CutPoint_t rest, candidate;
    //     rest.start = 0;
    //     rest.end = 10;
    //     candidate.start = 0;
    //     candidate.end = 5;
    //     float computed = evaluateCutPoint(rest, candidate);
    //     ASSERT_NEAR(0.468996, computed, precision_test);
    // }
    TEST_F(TestMetrics, ComputeCutPoints)
    {
        std::vector<CutPoint_t> computed, expected;
        computeCutPoints();
        computed = getCutPoints();
        for (auto cut : computed) {
            std::cout << "(" << cut.start << ", " << cut.end << ") -> (" << cut.fromValue << ",  " << cut.toValue << ")" << std::endl;
        }
    }
}