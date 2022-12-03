#include "gtest/gtest.h"
#include "../Metrics.h"
#include "../CPPFImdlp.h"
namespace mdlp {
    class TestMetrics : public CPPFImdlp, public testing::Test {
    public:
        TestMetrics() : CPPFImdlp(true, 6, true) {}
        indices_t indices; // sorted indices to use with X and y
        samples X;
        labels y;
        samples xDiscretized;
        int numClasses;
        float precision_test = 0.000001;
        void SetUp()
        {
            //    5.0, 5.1, 5.1, 5.1, 5.2, 5.3, 5.6, 5.7, 5.9, 6.0]
            //(5.0, 1) (5.1, 1) (5.1, 2) (5.1, 2) (5.2, 1) (5.3, 1) (5.6, 2) (5.7, 1) (5.9, 2) (6.0, 2)
            X = { 5.7, 5.3, 5.2, 5.1, 5.0, 5.6, 5.1, 6.0, 5.1, 5.9 };
            y = { 1, 1, 1, 1, 1, 2, 2, 2, 2, 2 };
            fit(X, y);
        }
        void check_sorted_vector(samples& X_, indices_t indices_)
        {
            X = X_;
            indices = indices_;
            indices_t testSortedIndices = sortIndices(X);
            float prev = X[testSortedIndices[0]];
            for (auto i = 0; i < X.size(); ++i) {
                EXPECT_EQ(testSortedIndices[i], indices[i]);
                EXPECT_LE(prev, X[testSortedIndices[i]]);
                prev = X[testSortedIndices[i]];
            }
        }
    };
    // 
    TEST_F(TestMetrics, SortIndices)
    {
        X = { 5.7, 5.3, 5.2, 5.1, 5.0, 5.6, 5.1, 6.0, 5.1, 5.9 };
        indices_t indices = { 4, 3, 6, 8, 2, 1, 5, 0, 9, 7 };
        check_sorted_vector(X, indices);
        X = { 5.77, 5.88, 5.99 };
        indices = { 0, 1, 2 };
        check_sorted_vector(X, indices);
        X = { 5.33, 5.22, 5.11 };
        indices = { 2, 1, 0 };
        check_sorted_vector(X, indices);
    }
    TEST_F(TestMetrics, EvaluateCutPoint)
    {
        cutPoint_t rest, candidate;
        rest.start = 0;
        rest.end = 10;
        rest.classNumber = -1;
        rest.fromValue = -1;
        rest.toValue = 1000;
        candidate.start = 0;
        candidate.end = 4;
        candidate.fromValue = -1;
        candidate.toValue = 5.15;
        candidate.classNumber = -1;
        EXPECT_FALSE(evaluateCutPoint(rest, candidate));
    }
    TEST_F(TestMetrics, ComputeCutPointsOriginal)
    {
        cutPoints_t computed, expected;
        expected = {
            { 0, 4, -1, -3.4028234663852886e+38, 5.15 }, { 4, 6, -1, 5.15, 5.45 },
            { 6, 7, -1, 5.45, 5.65 }, { 7, 10, -1, 5.65, 3.4028234663852886e+38 }
        };
        computeCutPointsOriginal();
        computed = getCutPoints();
        EXPECT_EQ(computed.size(), 4);
        for (auto i = 0; i < 4; i++) {
            EXPECT_EQ(computed[i].start, expected[i].start);
            EXPECT_EQ(computed[i].end, expected[i].end);
            EXPECT_EQ(computed[i].classNumber, expected[i].classNumber);
            EXPECT_NEAR(computed[i].fromValue, expected[i].fromValue, precision_test);
            EXPECT_NEAR(computed[i].toValue, expected[i].toValue, precision_test);
        }
    }
    TEST_F(TestMetrics, ComputeCutPointsOriginalGCase)
    {
        cutPoints_t computed, expected;
        expected = {
                { 0, 4, -1, -3.4028234663852886e+38, 3.4028234663852886e+38 },
        };
        X = { 0, 1, 2, 2 };
        y = { 1, 1, 1, 2 };
        fit(X, y);
        computeCutPointsOriginal();
        computed = getCutPoints();
        EXPECT_EQ(computed.size(), 1);
        for (auto i = 0; i < 1; i++) {
            EXPECT_EQ(computed[i].start, expected[i].start);
            EXPECT_EQ(computed[i].end, expected[i].end);
            EXPECT_EQ(computed[i].classNumber, expected[i].classNumber);
            EXPECT_NEAR(computed[i].fromValue, expected[i].fromValue, precision_test);
            EXPECT_NEAR(computed[i].toValue, expected[i].toValue, precision_test);
        }
    }
    TEST_F(TestMetrics, ComputeCutPointsProposed)
    {
        cutPoints_t computed, expected;
        expected = {
            { 0, 4, -1, -3.4028234663852886e+38, 5.1 }, { 4, 5, -1, 5.1, 5.2 },
            { 5, 6, -1, 5.2, 5.4 }, { 6, 9, -1, 5.4, 5.85 },
            { 9, 10, -1, 5.85, 3.4028234663852886e+38 }
        };
        computeCutPointsProposed();
        computed = getCutPoints();
        EXPECT_EQ(computed.size(), 5);
        for (auto i = 0; i < 5; i++) {
            EXPECT_EQ(computed[i].start, expected[i].start);
            EXPECT_EQ(computed[i].end, expected[i].end);
            EXPECT_EQ(computed[i].classNumber, expected[i].classNumber);
            EXPECT_NEAR(computed[i].fromValue, expected[i].fromValue, precision_test);
            EXPECT_NEAR(computed[i].toValue, expected[i].toValue, precision_test);
        }
    }
    TEST_F(TestMetrics, ComputeCutPointsProposedGCase)
    {
        cutPoints_t computed, expected;
        expected = {
                { 0, 3, -1, -3.4028234663852886e+38, 1.5 },
                { 3, 4, -1, 1.5, 3.4028234663852886e+38 }
        };
        X = { 0, 1, 2, 2 };
        y = { 1, 1, 1, 2 };
        fit(X, y);
        computeCutPointsProposed();
        computed = getCutPoints();
        EXPECT_EQ(computed.size(), 2);
        for (auto i = 0; i < 1; i++) {
            EXPECT_EQ(computed[i].start, expected[i].start);
            EXPECT_EQ(computed[i].end, expected[i].end);
            EXPECT_EQ(computed[i].classNumber, expected[i].classNumber);
            EXPECT_NEAR(computed[i].fromValue, expected[i].fromValue, precision_test);
            EXPECT_NEAR(computed[i].toValue, expected[i].toValue, precision_test);
        }
    }
    TEST_F(TestMetrics, ApplyCutPoints)
    {
        cutPoints_t expected = {
            { 0, 4, 17, -3.4028234663852886e+38, 5.1 }, { 4, 6, 31, 5.1, 5.4 },
            { 6, 8, 59, 5.4, 5.85 },
            { 8, 10, 41, 5.85, 3.4028234663852886e+38 }
        };
        setCutPoints(expected);
        applyCutPoints();
        labels expected_x = getDiscretizedValues();
        indices_t indices_x = getIndices();
        for (auto i = 0; i < 5; i++) {
            std::cout << "cutPoint[" << i << "].start = " << expected[i].start << std::endl;
            for (auto j = expected[i].start; j < expected[i].end; j++) {
                std::cout << expected_x[j] << expected[i].classNumber << std::endl;
                EXPECT_EQ(expected_x[indices_x[j]], expected[i].classNumber);
            }
        }
    }
}