#include "gtest/gtest.h"
#include "../Metrics.h"

namespace mdlp {
    precision_t precision = 0.000001;
    TEST(MetricTest, NumClasses)
    {
        labels y = { 1, 1, 1, 1, 1, 1, 1, 1, 2, 1 };
        indices_t indices = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        EXPECT_EQ(1, Metrics::numClasses(y, indices, 4, 8));
        EXPECT_EQ(2, Metrics::numClasses(y, indices, 0, 10));
        EXPECT_EQ(2, Metrics::numClasses(y, indices, 8, 10));
    }
    TEST(MetricTest, Entropy)
    {
        labels y = { 1, 1, 1, 1, 1, 2, 2, 2, 2, 2 };
        indices_t indices = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        EXPECT_EQ(1, Metrics::entropy(y, indices, 0, 10, 2));
        EXPECT_EQ(0, Metrics::entropy(y, indices, 0, 5, 1));
        labels yz = { 1, 1, 1, 1, 1, 1, 1, 1, 2, 1 };
        ASSERT_NEAR(0.468996, Metrics::entropy(yz, indices, 0, 10, 2), precision);
    }
    TEST(MetricTest, InformationGain)
    {
        labels y = { 1, 1, 1, 1, 1, 2, 2, 2, 2, 2 };
        indices_t indices = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        labels yz = { 1, 1, 1, 1, 1, 1, 1, 1, 2, 1 };
        ASSERT_NEAR(1, Metrics::informationGain(y, indices, 0, 10, 5, 2), precision);
        ASSERT_NEAR(0.108032, Metrics::informationGain(yz, indices, 0, 10, 5, 2), precision);
    }
}