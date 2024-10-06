#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include "Tensor.hpp"
#include "Softmax.hpp"
#include "Perceptron.hpp"

#define SIZE 500

int main()
{
    nn::Tensor m(SIZE, SIZE);
    nn::Tensor n(SIZE, SIZE);
    nn::Perceptron p(SIZE, SIZE);
    nn::Softmax s;
    std::ofstream outfile("oupt.txt");
    m.data = Eigen::MatrixXd::Random(SIZE, SIZE);
    outfile << m.data << std::endl;
    std::cout << "Start Compute" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    p.run(m, n);
    s.run(n, m);
    auto end = std::chrono::high_resolution_clock::now();
    outfile << m.data << std::endl;
    outfile.close();
    std::cout << "Time: " << std::chrono::duration<double>(end - start).count() << " s" << std::endl;
}