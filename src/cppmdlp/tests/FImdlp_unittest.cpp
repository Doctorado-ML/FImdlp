#include "gtest/gtest.h"
#include "../Metrics.h"
#include "../CPPFImdlp.h"
#include <iostream>

namespace mdlp {
    class TestFImdlp : public CPPFImdlp, public testing::Test {
    public:
        precision_t precision = 0.000001;

        TestFImdlp() : CPPFImdlp(false) {}

        void SetUp() {
            //    5.0, 5.1, 5.1, 5.1, 5.2, 5.3, 5.6, 5.7, 5.9, 6.0]
            //(5.0, 1) (5.1, 1) (5.1, 2) (5.1, 2) (5.2, 1) (5.3, 1) (5.6, 2) (5.7, 1) (5.9, 2) (6.0, 2)
            X = {5.7, 5.3, 5.2, 5.1, 5.0, 5.6, 5.1, 6.0, 5.1, 5.9};
            y = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2};
            proposal = false;
            fit(X, y);
        }

        void setProposal(bool value) {
            proposal = value;
        }

        // void initIndices()
        // {
        //     indices = indices_t();
        // }
        void checkSortedVector() {
            indices_t testSortedIndices = sortIndices(X);
            precision_t prev = X[testSortedIndices[0]];
            for (auto i = 0; i < X.size(); ++i) {
                EXPECT_EQ(testSortedIndices[i], indices[i]);
                EXPECT_LE(prev, X[testSortedIndices[i]]);
                prev = X[testSortedIndices[i]];
            }
        }

        void checkCutPoints(cutPoints_t &expected) {
            int expectedSize = expected.size();
            EXPECT_EQ(cutPoints.size(), expectedSize);
            for (auto i = 0; i < cutPoints.size(); i++) {
                EXPECT_NEAR(cutPoints[i], expected[i], precision);
            }
        }

        template<typename T, typename A>
        void checkVectors(std::vector<T, A> const &expected, std::vector<T, A> const &computed) {
            EXPECT_EQ(expected.size(), computed.size());
            ASSERT_EQ(expected.size(), computed.size());
            for (auto i = 0; i < expected.size(); i++) {
                EXPECT_NEAR(expected[i], computed[i],precision);
            }
        }
    };

    TEST_F(TestFImdlp, FitErrorEmptyDataset) {
        X = samples_t();
        y = labels_t();
        EXPECT_THROW(fit(X, y), std::invalid_argument);
    }

    TEST_F(TestFImdlp, FitErrorDifferentSize) {
        X = {1, 2, 3};
        y = {1, 2};
        EXPECT_THROW(fit(X, y), std::invalid_argument);
    }

    TEST_F(TestFImdlp, SortIndices) {
        X = {5.7, 5.3, 5.2, 5.1, 5.0, 5.6, 5.1, 6.0, 5.1, 5.9};
        indices = {4, 3, 6, 8, 2, 1, 5, 0, 9, 7};
        checkSortedVector();
        X = {5.77, 5.88, 5.99};
        indices = {0, 1, 2};
        checkSortedVector();
        X = {5.33, 5.22, 5.11};
        indices = {2, 1, 0};
        checkSortedVector();
    }

    TEST_F(TestFImdlp, TestDataset) {
        proposal = false;
        fit(X, y);
        computeCutPointsOriginal(0, 10);
        cutPoints_t expected = {5.6499996185302734};
        vector<precision_t> computed = getCutPoints();
        computed = getCutPoints();
        int expectedSize = expected.size();
        EXPECT_EQ(computed.size(), expected.size());
        for (auto i = 0; i < expectedSize; i++) {
            EXPECT_NEAR(computed[i], expected[i], precision);
        }
    }

    TEST_F(TestFImdlp, ComputeCutPointsOriginal) {
        cutPoints_t expected = {5.65};
        proposal = false;
        computeCutPointsOriginal(0, 10);
        checkCutPoints(expected);
    }

    TEST_F(TestFImdlp, ComputeCutPointsOriginalGCase) {
        cutPoints_t expected;
        proposal = false;
        expected = {2};
        samples_t X_ = {0, 1, 2, 2};
        labels_t y_ = {1, 1, 1, 2};
        fit(X_, y_);
        checkCutPoints(expected);
    }

    TEST_F(TestFImdlp, ComputeCutPointsProposal) {
        proposal = true;
        cutPoints_t expected;
        expected = {};
        fit(X, y);
        computeCutPointsProposal();
        checkCutPoints(expected);
    }

    TEST_F(TestFImdlp, ComputeCutPointsProposalGCase) {
        cutPoints_t expected;
        expected = {1.5};
        proposal = true;
        samples_t X_ = {0, 1, 2, 2};
        labels_t y_ = {1, 1, 1, 2};
        fit(X_, y_);
        checkCutPoints(expected);
    }

    TEST_F(TestFImdlp, GetCutPoints) {
        samples_t computed, expected = {5.65};
        proposal = false;
        computeCutPointsOriginal(0, 10);
        computed = getCutPoints();
        for (auto item: cutPoints)
            cout << setprecision(6) << item << endl;
        checkVectors(expected, computed);
    }
}
