#pragma once
#include "Tensor.hpp"
#include "BaseLayer.hpp"

namespace nn
{
    class Perceptron : public BaseLayer
    {
    public:
        Perceptron() = default;
        Perceptron(int input_size, int output_size) : weights(Eigen::MatrixXd::Random(input_size + 1, output_size)) {}
        void run(Tensor &in, Tensor &out) const override
        {
            Tensor bias = Tensor(in.data.rows() + 1, in.data.cols());
            bias.data << Eigen::MatrixXd::Ones(1, in.data.cols()), in.data;

            out.data = weights.data.transpose() * bias.data;
        };
        Tensor weights;
    };

}