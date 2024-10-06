#pragma once
#include <memory>
#include <Eigen/Dense>
#include <utility>

namespace nn
{
    class Tensor
    {
    public:
        Tensor() = default;
        Tensor(int rows, int cols) : data(rows, cols) {}
        template <typename T>
        Tensor(T&& data) : data(std::forward<T>(data)) {}

        Eigen::MatrixXd data;
    };
}