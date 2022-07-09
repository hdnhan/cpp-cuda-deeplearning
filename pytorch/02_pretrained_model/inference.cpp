#include <torch/script.h>
#include <torch/torch.h>

#include <chrono>
#include <iostream>

int main() {
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << device << std::endl;

    // Load the model
    torch::jit::script::Module net;
    net = torch::jit::load("./checkpoint/traced_model.pt");
    net.to(device);

    // Load the data
    auto testset = torch::data::datasets::MNIST(
                       "../data/MNIST/raw",
                       torch::data::datasets::MNIST::Mode::kTest)
                       .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                       .map(torch::data::transforms::Stack<>());
    // std::cout << testset.size().value() << std::endl;
    auto testloader = torch::data::make_data_loader<
        torch::data::samplers::SequentialSampler>(
        std::move(testset),
        torch::data::DataLoaderOptions().batch_size(64).workers(2));

    net.eval();
    torch::NoGradGuard no_grad;

    float total_loss = 0, total_acc = 0;
    int count_batch = 0, count_sample = 0;
    int cnt = 0, count = 0;

    auto start = std::chrono::high_resolution_clock::now();
    for (torch::data::Example<>& batch : *testloader) {
        std::vector<torch::jit::IValue> inputs;
        torch::Tensor data = batch.data.to(device);
        torch::Tensor labels = batch.target.to(device);
        inputs.push_back(data);

        torch::Tensor output = net.forward(inputs).toTensor();
        // torch::Tensor output = net.forward(data);
        total_acc += torch::argmax(output, 1).eq(labels).sum().item<int64_t>();
        torch::Tensor loss = torch::nn::functional::cross_entropy(output, labels);
        total_loss += loss.item<float>();
        count_batch += 1;
        count_sample += batch.data.size(0);
    }
    total_acc /= count_sample;
    total_loss /= count_batch;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    printf("test_loss: %.4f test_acc: %.4f time: %.4f\n", total_loss, total_acc, diff.count());
    return 0;
}
