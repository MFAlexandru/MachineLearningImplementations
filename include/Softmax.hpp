#pragma once
#include "Tensor.hpp"
#include "BaseLayer.hpp"

namespace nn
{
    class Softmax : public BaseLayer
    {
    public:
        void run(Tensor &in, Tensor &out) const override
        {
            // Numerically stable softmax: subtract max per row
            out.data = out.data - in.data.rowwise().maxCoeff().replicate(1, in.data.cols());

            // Apply the exponentials to each element
            out.data = out.data.array().exp();

            // Normalise the result
            out.data = out.data.array().colwise() / out.data.rowwise().sum().array();
        };
    };

}