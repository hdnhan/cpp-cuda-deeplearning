#include <torch/torch.h>

#include <iostream>

int main() {
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << device << std::endl;

    torch::Tensor tensor = torch::eye(3).to(device);
    std::cout << tensor << std::endl;

    return 0;
}