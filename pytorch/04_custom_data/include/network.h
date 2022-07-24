#pragma once

#include <torch/torch.h>

namespace network {
struct Net1Impl : public torch::nn::Module {
   private:
    torch::nn::Sequential backbone;

   public:
    Net1Impl();
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Net1);

class Net2Impl : public torch::nn::Module {
   private:
    torch::nn::Sequential backbone;

   public:
    Net2Impl();
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(Net2);
}  // namespace network