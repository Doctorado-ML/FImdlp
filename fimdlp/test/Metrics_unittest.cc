#include "gtest/gtest.h"
#include "../Metrics.h"
namespace
{
    TEST(MetricTest, NumClasses)
    {
        std::vector<int> y = {1, 1, 1, 1, 1, 1, 1, 1, 2, 1};
        std::vector<size_t> indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        EXPECT_EQ(1, mdlp::Metrics::numClasses(y, indices, 4, 8));
        EXPECT_EQ(2, mdlp::Metrics::numClasses(y, indices, 0, 10));
        EXPECT_EQ(2, mdlp::Metrics::numClasses(y, indices, 8, 10));
    }
    TEST(MetricTest, Entropy)
    {
        std::vector<int> y = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2};
        std::vector<size_t> indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        EXPECT_EQ(1, mdlp::Metrics::entropy(y, indices, 0, 10, 2));
        EXPECT_EQ(0, mdlp::Metrics::entropy(y, indices, 0, 5, 1));
        std::vector<int> yz = {1, 1, 1, 1, 1, 1, 1, 1, 2, 1};
        ASSERT_NEAR(0.468996, mdlp::Metrics::entropy(yz, indices, 0, 10, 2), 0.000001);
    }

}