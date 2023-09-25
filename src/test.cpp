#include "dataset.hpp"
#include "NN.hpp"
#include "../lib/matrix.h"
#include <vector>
#include <iostream>
#include <cassert>
#include "gtest/gtest.h"


TEST(FunctionTesting, sigmoid_test) {
	std::vector<double> testVector1{ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
	std::vector<double> expectVector1{ 0.5, 0.731059, 0.880797, 0.952574, 0.982014, 0.993307, 0.997527, 0.999089, 0.999665, 0.999877 };

	std::vector<double> testVector2{ 3123123.0, 124534124.0, 0.0, 0.0, 123123.0 };
	std::vector<double> expectVector2{ 1, 1, 0.5, 0.5, 1 };

	std::vector<double> testVector3;
	std::vector<double> expectVector3;

	EXPECT_NEAR(NeuralNetwork::sigmoid(testVector1), expectVector1, 1e-6);

	EXPECT_NEAR(NeuralNetwork::sigmoid(testVector2), expectVector2, 1e-6);

	EXPECT_NEAR(NeuralNetwork::sigmoid(testVector3), expectVector3, 1e-6);
}

TEST(FunctionTesting, sigmoid_prime_test) {
	std::vector<double> testVector1{ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
	std::vector<double> expectVector1{ 0.25, 0.196612, 0.104994, 0.0451767, 0.0176627, 0.00664806, 0.00246651, 0.000910221, 0.000335238, 0.000123379 };

	std::vector<double> testVector2{ 1.0, 100.0, 300.0, 2.0, 34.0 };
	std::vector<double> expectVector2{ 0.196612, 3.72008e-44, 5.1482e-131, 0.104994, 1.71391e-15 };

	std::vector<double> testVector3;
	std::vector<double> expectVector3;

	EXPECT_NEAR(NeuralNetwork::sigmoid_prime(testVector1), expectVector1, 1e-6);

	EXPECT_NEAR(NeuralNetwork::sigmoid_prime(testVector2), expectVector2, 1e-6);

	EXPECT_NEAR(NeuralNetwork::sigmoid(testVector3), expectVector3, 1e-6);
}

TEST(FunctionTesting, log_test) {
	std::vector<double> testVector1{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
	std::vector<double> expectVector1{0, 0.693147, 1.09861, 1.38629, 1.609440, 1.79176, 1.94591, 2.07944, 2.19722  };

	std::vector<double> testVector2{ 1.0, 100.0, 300.0, 2.0, 34.0 };
	std::vector<double> expectVector2{ 0, 4.60517, 5.70378, 0.693147, 3.52636 };

	std::vector<double> testVector3;
	std::vector<double> expectVector3;

	EXPECT_NEAR(log(testVector1), expectVector1, 1e-6);

	EXPECT_NEAR(log(testVector2), expectVector2, 1e-6);

	EXPECT_NEAR(log(testVector3), expectVector3, 1e-6);
}


int main(int argc, char** argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}