
#include "dataset.hpp"
#include "NN.hpp"
#include "../lib/matrix.h"
#include <vector>
#include <iostream>

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

//Начало тестов

#ifdef TESTS
#include <gtest/gtest.h>

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
    std::vector<double> x1 = {0.1, 0.5, 0.8, 0.02, 0.95};
    x1_true = {0.1, 0.5, 0.8, 0.02, 0.95};
    ASSERT_EQ(LeakyReLU(x1),x1_true);
}

TEST(FunctionTesting, testLeakyReLUMixed){
    std::vector<double> x2 = {-0.61, 0.45, -0.71, 0.12, 0.99};
    std::vector<double> x2_true = {-0.61*0.01, 0.45, -0.71*0.01, 0.12, 0.99};
    ASSERT_EQ(LeakyReLU(x2),x2_true);
}

TEST(FunctionTesting, testLeakyReLUNegatives){
    std::vector<double> x3 = {-0.32, -0.64, -0.56, -0.57, -0.91};
    std::vector<double> x3_true = {-0.32*0.01, -0.64*0.01, -0.56*0.01, -0.57*0.01, -0.91*0.01};
    ASSERT_EQ(LeakyReLU(x3),x3_true);
}
#endif

int main(int argc, char **argv) {

    #ifdef TESTS
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    #endif
    
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
    
    return 0;
}
