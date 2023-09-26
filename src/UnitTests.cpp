#include <iostream>
#include <math.h>
#include "gtest/gtest.h"

std::vector<double> x1 = {0.1, 0.5, 0.8, 0.02, 0.95};
std::vector<double> x1_true = {0.1, 0.5, 0.8, 0.02, 0.95}
std::vector<double> x2 = {-0.61, 0.45, -0.71, 0.12, 0.99};
std::vector<double> x2_true = {-0.0061, 0.45, -0.0071, 0.12, 0.99}
std::vector<double> x3 = {-0.32, -0.64, -0.56, -0.57, -0.91};
std::vector<double> x3_true = {-0.0032, -0.0064, -0.0056, -0.0057, -0.0091};
  
double max(double a, double b) {
    if(a >= b)
       return a;
    else
       return b;
}

std::vector<double> LeakyReLU(const std::vector<double>& x) {
    std::vector<double> result(x.size());
    for (unsigned int i = 0; i < x.size(); i++)
        result[i] = max(x[i], 0.01*x[i]);
    return result;
}
  
TEST(FunctionTesting, testMax1){
  EXPECT_NEAR(max(0.1, 0.5),0.5,1e-6);
  EXPECT_NEAR(max(-0.1, -0.5),-0.1,1e-6);
  EXPECT_NEAR(max(0, 0.2),0.2,1e-6);
}

TEST(FunctionTesting, testMax2){
  EXPECT_NEAR(max(0.1, 0.01 * 0.1),0.1,1e-6);
  EXPECT_NEAR(max(-0.1, -0.01 * 0.1),-0.001,1e-6);
  EXPECT_NEAR(max(0, 0.01 * 0),0,1e-6);
}

TEST(FunctionTesting, testLeakyReLUPositives){
  ASSERT_EQ(LeakyReLU(x1),x1_true);
}

TEST(FunctionTesting, testLeakyReLUMixed){
  ASSERT_EQ(LeakyReLU(x2),x2_true);
}

TEST(FunctionTesting, testLeakyReLUNegatives){
  ASSERT_EQ(LeakyReLU(x3),x3_true);
}

int main(int argc, char **argv)
{
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
