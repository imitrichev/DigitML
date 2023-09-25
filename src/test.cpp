#include <vector>
#include <iostream>
#include "gtest/gtest.h"

std::vector<double> sigmoid(const std::vector<double>& x) {
	std::vector<double> result(x.size());
	for (unsigned int i = 0; i < x.size(); i++)
		result[i] = 1 / (1 + exp(-x[i]));
	return result;
}

std::vector<double> sigmoid_prime(const std::vector<double>& x) {
	std::vector<double> result(x.size());
	for (unsigned int i = 0; i < result.size(); i++) {
		const double t = exp(x[i]);
		result[i] = t / ((1 + t) * (1 + t));
	}
	return result;
}

std::vector<double> vectorize_label(unsigned char label) {
	std::vector<double> result(10, 0.0);
	result[(unsigned int)label] = 1.0;
	return result;
}

std::vector<double> log(const std::vector<double>& vec) {
	std::vector<double> result(vec.size());
	for (unsigned int i = 0; i < result.size(); ++i) {
		// Currently not checking for log(0) errors, but it seems fine
		result[i] = log(vec[i]);
	}
	return result;
}

TEST(FunctionTesting, sigmoid_test) {

	std::vector<double> expectVector { 0.5, 0.731059, 0.880797, 0.999955, 1, 1};
	std::vector<double> testVector { 0.0, 1.0, 2.0, 10.0, 100.0, 10000.0};
	std::vector<double> result = sigmoid(testVector);
	
	for (size_t i = 0; i < testVector.size(); i++) {
		EXPECT_NEAR(result[i], expectVector[i], 1e-6);
	}

}

TEST(FunctionTesting, sigmoid_prime_test) {

	std::vector<double> expectVector { 0.25, 0.196612, 0.104994, 4.53958e-05, 3.72008e-44};
	std::vector<double> testVector { 0.0, 1.0, 2.0, 10.0, 100.0};
	std::vector<double> result = sigmoid_prime(testVector);
	
	for (size_t i = 0; i < testVector.size(); i++) {
		EXPECT_NEAR(result[i], expectVector[i], 1e-6);
	}

}

TEST(FunctionTesting, log_test) {

	std::vector<double> expectVector { 0, 0.693147, 2.30259, 4.60517, 9.21034 };
	std::vector<double> testVector { 0.0, 1.0, 2.0, 10.0, 100.0 };
	std::vector<double> result = log(testVector);
	
	for (size_t i = 0; i < testVector.size(); i++) {
		EXPECT_NEAR(result[i], expectVector[i], 1e-6);
	}

}

TEST(FunctionTesting, vectorize_label_test) {

	std::vector<double> expectVector0{ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
	std::vector<double> expectVector1{ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
	std::vector<double> expectVector2{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
	
	std::vector<double> result0 = vectorize_label(0);
	std::vector<double> result1 = vectorize_label(1);
	std::vector<double> result2 = vectorize_label(15);

	for (size_t i = 0; i < expectVector0.size(); i++) {
		EXPECT_NEAR(result0[i], expectVector0[i], 1e-6);
	}

	for (size_t i = 0; i < expectVector1.size(); i++) {
		EXPECT_NEAR(result1[i], expectVector1[i], 1e-6);
	}

	for (size_t i = 0; i < expectVector2.size(); i++) {
		EXPECT_NEAR(result2[i], expectVector2[i], 1e-6);
	}
}


int main(int argc, char** argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}