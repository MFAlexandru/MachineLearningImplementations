#pragma once
#include <memory>
#include <vector>
#include "Tensor.hpp"

namespace nn
{
    class BaseLayer
    {
    public:
        virtual void run(Tensor &in, Tensor &out) const = 0;
    };
}