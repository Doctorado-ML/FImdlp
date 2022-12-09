//#include "gtest/gtest.h"
//#include "../Metrics.h"
//#include "../CPPFImdlp.h"
//namespace mdlp {
//    class TestFImdlp : public CPPFImdlp, public testing::Test {
//    public:
//        TestFImdlp() : CPPFImdlp(true, true) {}
//        void SetUp()
//        {
//            //    5.0, 5.1, 5.1, 5.1, 5.2, 5.3, 5.6, 5.7, 5.9, 6.0]
//            //(5.0, 1) (5.1, 1) (5.1, 2) (5.1, 2) (5.2, 1) (5.3, 1) (5.6, 2) (5.7, 1) (5.9, 2) (6.0, 2)
//            X = { 5.7, 5.3, 5.2, 5.1, 5.0, 5.6, 5.1, 6.0, 5.1, 5.9 };
//            y = { 1, 1, 1, 1, 1, 2, 2, 2, 2, 2 };
//            fit(X, y);
//        }
//        void setProposal(bool value)
//        {
//            proposal = value;
//        }
//        void initCutPoints()
//        {
//            setCutPoints(cutPoints_t());
//        }
//        void initIndices()
//        {
//            indices = indices_t();
//        }
//        void initDiscretized()
//        {
//            xDiscretized = labels();
//        }
//        void checkSortedVector(samples& X_, indices_t indices_)
//        {
//            X = X_;
//            indices = indices_;
//            indices_t testSortedIndices = sortIndices(X);
//            precision_t prev = X[testSortedIndices[0]];
//            for (auto i = 0; i < X.size(); ++i) {
//                EXPECT_EQ(testSortedIndices[i], indices[i]);
//                EXPECT_LE(prev, X[testSortedIndices[i]]);
//                prev = X[testSortedIndices[i]];
//            }
//        }
//        void checkCutPoints(cutPoints_t& expected)
//        {
//            int expectedSize = expected.size();
//            EXPECT_EQ(cutPoints.size(), expectedSize);
//            for (auto i = 0; i < expectedSize; i++) {
//                EXPECT_EQ(cutPoints[i].start, expected[i].start);
//                EXPECT_EQ(cutPoints[i].end, expected[i].end);
//                EXPECT_EQ(cutPoints[i].classNumber, expected[i].classNumber);
//                EXPECT_NEAR(cutPoints[i].fromValue, expected[i].fromValue, precision);
//                EXPECT_NEAR(cutPoints[i].toValue, expected[i].toValue, precision);
//            }
//        }
//        template<typename T, typename A>
//        void checkVectors(std::vector<T, A> const& expected, std::vector<T, A> const& computed)
//        {
//            EXPECT_EQ(expected.size(), computed.size());
//            for (auto i = 0; i < expected.size(); i++) {
//                EXPECT_EQ(expected[i], computed[i]);
//            }
//        }
//
//    };
//    TEST_F(TestFImdlp, FitErrorEmptyDataset)
//    {
//        X = samples();
//        y = labels();
//        EXPECT_THROW(fit(X, y), std::invalid_argument);
//    }
//    TEST_F(TestFImdlp, FitErrorDifferentSize)
//    {
//        X = { 1, 2, 3 };
//        y = { 1, 2 };
//        EXPECT_THROW(fit(X, y), std::invalid_argument);
//    }
//    TEST_F(TestFImdlp, SortIndices)
//    {
//        X = { 5.7, 5.3, 5.2, 5.1, 5.0, 5.6, 5.1, 6.0, 5.1, 5.9 };
//        indices = { 4, 3, 6, 8, 2, 1, 5, 0, 9, 7 };
//        checkSortedVector(X, indices);
//        X = { 5.77, 5.88, 5.99 };
//        indices = { 0, 1, 2 };
//        checkSortedVector(X, indices);
//        X = { 5.33, 5.22, 5.11 };
//        indices = { 2, 1, 0 };
//        checkSortedVector(X, indices);
//    }
//    TEST_F(TestFImdlp, EvaluateCutPoint)
//    {
//        cutPoint_t rest, candidate;
//        rest = { 0, 10, -1, -1, 1000 };
//        candidate = { 0, 4, -1, -1, 5.15 };
//        EXPECT_FALSE(evaluateCutPoint(rest, candidate));
//    }
//    TEST_F(TestFImdlp, ComputeCutPointsOriginal)
//    {
//        cutPoints_t  expected;
//        expected = {
//            { 0, 4, -1, -3.4028234663852886e+38, 5.15 }, { 4, 6, -1, 5.15, 5.45 },
//            { 6, 10, -1, 5.45, 3.4028234663852886e+38 }
//        };
//        setCutPoints(cutPoints_t());
//        computeCutPointsOriginal();
//        checkCutPoints(expected);
//    }
//    TEST_F(TestFImdlp, ComputeCutPointsOriginalGCase)
//    {
//        cutPoints_t  expected;
//        expected = {
//                { 0, 4, -1, -3.4028234663852886e+38, 3.4028234663852886e+38 },
//        };
//        X = { 0, 1, 2, 2 };
//        y = { 1, 1, 1, 2 };
//        fit(X, y);
//        computeCutPointsOriginal();
//        checkCutPoints(expected);
//    }
//    TEST_F(TestFImdlp, ComputeCutPointsProposal)
//    {
//        cutPoints_t  expected;
//        expected = {
//            { 0, 4, -1, -3.4028234663852886e+38, 5.1 }, { 4, 6, -1, 5.1, 5.4 },
//            { 6, 9, -1, 5.4, 5.85 },
//            { 9, 10, -1, 5.85, 3.4028234663852886e+38 }
//        };
//        computeCutPointsProposal();
//        checkCutPoints(expected);
//    }
//    TEST_F(TestFImdlp, ComputeCutPointsProposalGCase)
//    {
//        cutPoints_t  expected;
//        expected = {
//                { 0, 3, -1, -3.4028234663852886e+38, 1.5 },
//                { 3, 4, -1, 1.5, 3.4028234663852886e+38 }
//        };
//        X = { 0, 1, 2, 2 };
//        y = { 1, 1, 1, 2 };
//        fit(X, y);
//        computeCutPointsProposal();
//        checkCutPoints(expected);
//    }
//    TEST_F(TestFImdlp, DiscretizedValues)
//    {
//        labels computed, expected = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
//        computed = getDiscretizedValues();
//        checkVectors(expected, computed);
//    }
//    TEST_F(TestFImdlp, GetCutPoints)
//    {
//        samples computed, expected = { 5.15, 5.45, 3.4028234663852886e+38 };
//        computeCutPointsOriginal();
//        computed = getCutPoints();
//        checkVectors(expected, computed);
//    }
//    TEST_F(TestFImdlp, Constructor)
//    {
//        samples X = { 5.7, 5.3, 5.2, 5.1, 5.0, 5.6, 5.1, 6.0, 5.1, 5.9 };
//        labels y = { 1, 1, 1, 1, 1, 2, 2, 2, 2, 2 };
//        setProposal(false);
//        fit(X, y);
//        computeCutPointsOriginal();
//        cutPoints_t expected;
//        vector<precision_t> computed = getCutPoints();
//        expected = {
//            { 0, 4, -1, -3.4028234663852886e+38, 5.15 }, { 4, 6, -1, 5.15, 5.45 },
//            { 6, 10, -1, 5.45, 3.4028234663852886e+38 }
//        };
//        computed = getCutPoints();
//        int expectedSize = expected.size();
//        EXPECT_EQ(computed.size(), expected.size());
//        for (auto i = 0; i < expectedSize; i++) {
//            EXPECT_NEAR(computed[i], expected[i].toValue, .00000001);
//        }
//    }
//}