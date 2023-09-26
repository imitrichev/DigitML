
#include "dataset.hpp"
#include "NN.hpp"
#include "../lib/matrix.h"
#include <vector>
#include <iostream>
#include <ctime>

void debug(Example e) {
    static std::string shades = " .:-=+*#%@";
    for (unsigned int i = 0; i < 28 * 28; i++) {
        if (i % 28 == 0) printf("\n");
        printf("%c", shades[e.data[i] / 30]);
    }
    printf("\nLabel: %d\n", e.label);
}

std::vector<double> load_matrix(Example& e) {
    std::vector<double> result(e.data, e.data + 28 * 28);
    return result;
}

const double calculate_accuracy(const Matrix<unsigned char>& images, const Matrix<unsigned char>& labels, NeuralNetwork n) {
  unsigned int correct = 0;
  for (unsigned int i = 0; i < images.rows(); ++i) {
    Example e;
    for (int j = 0; j < 28*28; ++j) {
        e.data[j] = images[i][j];
    }
    e.label = labels[i][0];
    unsigned int guess = n.compute(e);
    if (guess == (unsigned int)e.label) correct++;
  }
  const double accuracy = (double)correct/images.rows();

  return accuracy;
}


#ifdef TESTS
#include "gtest/gtest.h"

double prelu(double x) {
    double alpha = 1.2;
    if (x >= 0) {
        return x;
    }
    else {
        return alpha * x;
    }
}

std::vector<double> PReLU(const std::vector<double>& x) {
    std::vector<double> result(x.size());
    for (unsigned int i = 0; i < result.size(); i++) {
        result[i] = prelu(x[i]);
    }
    return result;
}
TEST(FunctionTesting, testPrelu1) {
    EXPECT_NEAR(prelu(0.29), 0.29, 1e-6);
    EXPECT_NEAR(prelu(-0.92), -1.104, 1e-6);
    EXPECT_NEAR(prelu(0.83), 0.83, 1e-6);
}

TEST(FunctionTesting, testPrelu2) {
    EXPECT_NEAR(prelu(0.01), 0.01, 1e-6);
    EXPECT_NEAR(prelu(-0.01), -0.012, 1e-6);
    EXPECT_NEAR(prelu(0), 0.0, 1e-6);
}

TEST(FunctionTesting, testPReLUPos) {
    std::vector<double> x1 = { 0.16, 0.54, 0.23, 0.46, 0,563 };
    std::vector<double> right_x1 = { 0.16, 0.54, 0.23, 0.46, 0.563 };
    ASSERT_EQ(PReLU(x1), right_x1);
}

TEST(FunctionTesting, testPReLUMix) {
    std::vector<double> x2 = { 0.05, -0.45, -0.24, 0.01, -0.99 };
    std::vector<double> right_x2 = { 0.05, -0.54, -0.288, 0.01, -1.188 };
    ASSERT_EQ(PReLU(x2), right_x2);
}

TEST(FunctionTesting, testPReLUNeg) {
    std::vector<double> x3 = { -0.75, -0.93, -0.38, -0.02, -0.63 };
    std::vector<double> right_x3 = { -0.9, -1.116, -0.456, -0.024, -0.756 };
    ASSERT_EQ(PReLU(x3), right_x3);
}

#endif


int main(int argc, char** argv) {
    srand(time(NULL));
    Matrix<unsigned char> images_train(0, 0);
    Matrix<unsigned char> labels_train(0, 0);
    load_dataset(images_train, labels_train, "data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");

    Matrix<unsigned char> images_test(0, 0);
    Matrix<unsigned char> labels_test(0, 0);
    load_dataset(images_test, labels_test, "data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");

    NeuralNetwork n;

    const unsigned int num_iterations = 5;
    n.train(num_iterations, images_train, labels_train);

    const double accuracy_train = calculate_accuracy(images_train, labels_train, n);
    const double accuracy_test = calculate_accuracy(images_test, labels_test, n);

    printf("Accuracy on training data: %f\n", accuracy_train);
    printf("Accuracy on test data: %f\n", accuracy_test);

    #ifdef TESTS
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    #endif
    
    return 0;
}
