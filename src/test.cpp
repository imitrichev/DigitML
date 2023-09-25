#include "dataset.hpp"
#include "NN.hpp"
#include "../lib/matrix.h"
#include <vector>
#include <iostream>
#include <cassert>
#include "gtest/gtest.h"


TEST(FunctionTesting, sigmoid_test) {

	EXPECT_NEAR(sigmoid(0.0), 0.5, 1e-6);
	EXPECT_NEAR(sigmoid(1.0), 0.731059, 1e-6);
	EXPECT_NEAR(sigmoid(2.0), 0.880797, 1e-6);
	EXPECT_NEAR(sigmoid(3123123.0), 1, 1e-6);
	EXPECT_NEAR(sigmoid(124534124.0), 1, 1e-6);
	EXPECT_NEAR(sigmoid(5.0), 0.993307, 1e-6);
	EXPECT_NEAR(sigmoid(nullptr), nullptr, 1e-6);

}

TEST(FunctionTesting, sigmoid_prime_test) {

	EXPECT_NEAR(sigmoid_prime(0.0), 0.25, 1e-6);
	EXPECT_NEAR(sigmoid_prime(1.0), 0.196612, 1e-6);
	EXPECT_NEAR(sigmoid_prime(2.0), 0.104994, 1e-6);
	EXPECT_NEAR(sigmoid_prime(3.0), 0.0451767, 1e-6);
	EXPECT_NEAR(sigmoid_prime(4.0), 0.0176627, 1e-6);
	EXPECT_NEAR(sigmoid_prime(5.0), 0.00664806, 1e-6);
	EXPECT_NEAR(sigmoid_prime(6.0), 0.00246651, 1e-6);
	EXPECT_NEAR(sigmoid_prime(7.0), 0.000910221, 1e-6);
	EXPECT_NEAR(sigmoid_prime(8.0), 0.000335238, 1e-6);
	EXPECT_NEAR(sigmoid_prime(9.0), 0.000123379, 1e-6);
	EXPECT_NEAR(sigmoid_prime(100.0), 3.72008e-44, 1e-6);
	EXPECT_NEAR(sigmoid_prime(34.0), 1.71391e-15, 1e-6);

	EXPECT_NEAR(sigmoid_prime(nullptr), nullptr, 1e-6);
}

TEST(FunctionTesting, log_test) {

	EXPECT_NEAR(log(1.0), 0, 1e-6);
	EXPECT_NEAR(log(2.0), 0.693147, 1e-6);
	EXPECT_NEAR(log(3.0), 1.09861, 1e-6);
	EXPECT_NEAR(log(4.0), 1.38629, 1e-6);
	EXPECT_NEAR(log(5.0), 1.609440, 1e-6);
	EXPECT_NEAR(log(6.0), 1.79176, 1e-6);
	EXPECT_NEAR(log(7.0), 1.94591, 1e-6);
	EXPECT_NEAR(log(8.0), 2.07944, 1e-6);
	EXPECT_NEAR(log(9.0), 2.19722, 1e-6);
	EXPECT_NEAR(log(100.0), 4.60517, 1e-6);
	EXPECT_NEAR(log(300.0), 0.693147, 1e-6);
	EXPECT_NEAR(log(nullptr), nullptr, 1e-6);

}


int main(int argc, char** argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}