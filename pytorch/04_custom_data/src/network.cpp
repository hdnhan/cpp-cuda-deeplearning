#include "network.h"

namespace network {

Net1Impl::Net1Impl() : backbone(
                           torch::nn::Conv2d(3, 16, 3),
                           torch::nn::ReLU(),
                           torch::nn::MaxPool2d(2),
                           torch::nn::Conv2d(16, 32, 3),
                           torch::nn::ReLU(),
                           torch::nn::MaxPool2d(2),
                           torch::nn::Conv2d(32, 64, 3),
                           torch::nn::ReLU(),
                           torch::nn::AdaptiveMaxPool2d(5),
                           torch::nn::Flatten(),
                           torch::nn::Linear(64 * 5 * 5, 512),
                           torch::nn::ReLU(),
                           torch::nn::Linear(512, 2)) {
    register_module("backbone", backbone);
}

torch::Tensor Net1Impl::forward(torch::Tensor x) {
    return backbone->forward(x);
}

Net2Impl::Net2Impl() : backbone(torch::nn::Conv2d(3, 16, 3),
                                torch::nn::ReLU(),
                                torch::nn::MaxPool2d(2),
                                torch::nn::Conv2d(16, 32, 3),
                                torch::nn::ReLU(),
                                torch::nn::MaxPool2d(2),
                                torch::nn::Conv2d(32, 64, 3),
                                torch::nn::ReLU(),
                                torch::nn::AdaptiveMaxPool2d(5),
                                torch::nn::Flatten(),
                                torch::nn::Linear(64 * 5 * 5, 512),
                                torch::nn::ReLU(),
                                torch::nn::Linear(512, 2)) {
    register_module("backbone", backbone);
}

torch::Tensor Net2Impl::forward(torch::Tensor x) {
    x = backbone->forward(x);
    return x;
}

}  // namespace network