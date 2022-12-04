#include "gtest/gtest.h"
#include "../Metrics.h"
#include "../CPPFImdlp.h"
namespace mdlp {
    class TestFImdlp : public CPPFImdlp, public testing::Test {
    public:
        TestFImdlp() : CPPFImdlp(true, 6, true) {}
        void SetUp()
        {
            //    5.0, 5.1, 5.1, 5.1, 5.2, 5.3, 5.6, 5.7, 5.9, 6.0]
            //(5.0, 1) (5.1, 1) (5.1, 2) (5.1, 2) (5.2, 1) (5.3, 1) (5.6, 2) (5.7, 1) (5.9, 2) (6.0, 2)
            X = { 5.7, 5.3, 5.2, 5.1, 5.0, 5.6, 5.1, 6.0, 5.1, 5.9 };
            y = { 1, 1, 1, 1, 1, 2, 2, 2, 2, 2 };
            fit(X, y);
        }
        void initCutPoints()
        {
            setCutPoints(cutPoints_t());
        }
        void initIndices()
        {
            indices = indices_t();
        }
        void initDiscretized()
        {
            xDiscretized = labels();
        }
        void checkSortedVector(samples& X_, indices_t indices_)
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
    TEST_F(TestFImdlp, SortIndices)
    {
        X = { 5.7, 5.3, 5.2, 5.1, 5.0, 5.6, 5.1, 6.0, 5.1, 5.9 };
        indices = { 4, 3, 6, 8, 2, 1, 5, 0, 9, 7 };
        checkSortedVector(X, indices);
        X = { 5.77, 5.88, 5.99 };
        indices = { 0, 1, 2 };
        checkSortedVector(X, indices);
        X = { 5.33, 5.22, 5.11 };
        indices = { 2, 1, 0 };
        checkSortedVector(X, indices);
    }
    TEST_F(TestFImdlp, EvaluateCutPoint)
    {
        cutPoint_t rest, candidate;
        rest = { 0, 10, -1, -1, 1000 };
        candidate = { 0, 4, -1, -1, 5.15 };
        EXPECT_FALSE(evaluateCutPoint(rest, candidate));
    }
    TEST_F(TestFImdlp, ComputeCutPointsOriginal)
    {
        cutPoints_t computed, expected;
        int expectedSize = 3;
        expected = {
            { 0, 4, -1, -3.4028234663852886e+38, 5.15 }, { 4, 6, -1, 5.15, 5.45 },
            { 6, 10, -1, 5.45, 3.4028234663852886e+38 }
        };
        setCutPoints(cutPoints_t());
        computeCutPointsOriginal();
        computed = getCutPoints();
        EXPECT_EQ(computed.size(), expectedSize);
        for (auto i = 0; i < expectedSize; i++) {
            EXPECT_EQ(computed[i].start, expected[i].start);
            EXPECT_EQ(computed[i].end, expected[i].end);
            EXPECT_EQ(computed[i].classNumber, expected[i].classNumber);
            EXPECT_NEAR(computed[i].fromValue, expected[i].fromValue, precision);
            EXPECT_NEAR(computed[i].toValue, expected[i].toValue, precision);
        }
    }
    TEST_F(TestFImdlp, ComputeCutPointsOriginalGCase)
    {
        cutPoints_t computed, expected;
        expected = {
                { 0, 4, -1, -3.4028234663852886e+38, 3.4028234663852886e+38 },
        };
        int expectedSize = 1;
        X = { 0, 1, 2, 2 };
        y = { 1, 1, 1, 2 };
        fit(X, y);
        computeCutPointsOriginal();
        computed = getCutPoints();
        EXPECT_EQ(computed.size(), expectedSize);
        for (auto i = 0; i < expectedSize; i++) {
            EXPECT_EQ(computed[i].start, expected[i].start);
            EXPECT_EQ(computed[i].end, expected[i].end);
            EXPECT_EQ(computed[i].classNumber, expected[i].classNumber);
            EXPECT_NEAR(computed[i].fromValue, expected[i].fromValue, precision);
            EXPECT_NEAR(computed[i].toValue, expected[i].toValue, precision);
        }
    }
    TEST_F(TestFImdlp, ComputeCutPointsProposed)
    {
        cutPoints_t computed, expected;
        expected = {
            { 0, 4, -1, -3.4028234663852886e+38, 5.1 }, { 4, 6, -1, 5.1, 5.4 },
            { 6, 9, -1, 5.4, 5.85 },
            { 9, 10, -1, 5.85, 3.4028234663852886e+38 }
        };
        int expectedSize = 4;
        computeCutPointsProposed();
        computed = getCutPoints();
        EXPECT_EQ(computed.size(), expectedSize);
        for (auto i = 0; i < expectedSize; i++) {
            EXPECT_EQ(computed[i].start, expected[i].start);
            EXPECT_EQ(computed[i].end, expected[i].end);
            EXPECT_EQ(computed[i].classNumber, expected[i].classNumber);
            EXPECT_NEAR(computed[i].fromValue, expected[i].fromValue, precision);
            EXPECT_NEAR(computed[i].toValue, expected[i].toValue, precision);
        }
    }
    TEST_F(TestFImdlp, ComputeCutPointsProposedGCase)
    {
        cutPoints_t computed, expected;
        expected = {
                { 0, 3, -1, -3.4028234663852886e+38, 1.5 },
                { 3, 4, -1, 1.5, 3.4028234663852886e+38 }
        };
        int expectedSize = 2;
        X = { 0, 1, 2, 2 };
        y = { 1, 1, 1, 2 };
        fit(X, y);
        computeCutPointsProposed();
        computed = getCutPoints();
        EXPECT_EQ(computed.size(), expectedSize);
        for (auto i = 0; i < expectedSize; i++) {
            EXPECT_EQ(computed[i].start, expected[i].start);
            EXPECT_EQ(computed[i].end, expected[i].end);
            EXPECT_EQ(computed[i].classNumber, expected[i].classNumber);
            EXPECT_NEAR(computed[i].fromValue, expected[i].fromValue, precision);
            EXPECT_NEAR(computed[i].toValue, expected[i].toValue, precision);
        }
    }
    TEST_F(TestFImdlp, ApplyCutPoints)
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