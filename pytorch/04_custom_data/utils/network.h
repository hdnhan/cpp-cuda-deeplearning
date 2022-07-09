#pragma once

#include <torch/torch.h>

struct ConvNet1Impl : torch::nn::Module {
    ConvNet1Impl() : backbone(
                         torch::nn::Conv2d(3, 16, 3),
                         torch::nn::ReLU(),
                         torch::nn::MaxPool2d(2),
                         torch::nn::Conv2d(16, 32, 3),
                         torch::nn::ReLU(),
                         torch::nn::MaxPool2d(2),
                         torch::nn::Conv2d(32, 64, 3),
                         torch::nn::ReLU(),
                         //  torch::nn::MaxPool2d(2),
                         torch::nn::AdaptiveMaxPool2d(5),
                         torch::nn::Flatten(),
                         torch::nn::Linear(64 * 5 * 5, 512),
                         torch::nn::ReLU(),
                         torch::nn::Linear(512, 2)) {
        register_module("backbone", backbone);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = backbone->forward(x);
        return x;
    }
    torch::nn::Sequential backbone;
};
TORCH_MODULE(ConvNet1);

class ConvNet2Impl : public torch::nn::Module {
   private:
    torch::nn::Sequential backbone{
        torch::nn::Conv2d(3, 16, 3),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(2),
        torch::nn::Conv2d(16, 32, 3),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(2),
        torch::nn::Conv2d(32, 64, 3),
        torch::nn::ReLU(),
        //  torch::nn::MaxPool2d(2),
        torch::nn::AdaptiveMaxPool2d(5),
        torch::nn::Flatten(),
        torch::nn::Linear(64 * 5 * 5, 512),
        torch::nn::ReLU(),
        torch::nn::Linear(512, 2)};

   public:
    ConvNet2Impl() {
        register_module("backbone", backbone);
    }
    torch::Tensor forward(torch::Tensor x) {
        x = backbone->forward(x);
        return x;
    }
};

TORCH_MODULE(ConvNet2);