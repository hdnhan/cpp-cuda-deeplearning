#include <torch/torch.h>

#include <chrono>
#include <iostream>

int main() {
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << device << std::endl;

    // Define model
    torch::nn::Sequential net(
        torch::nn::Conv2d(1, 6, 5),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(2),
        torch::nn::Conv2d(6, 16, 5),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(2),
        torch::nn::Flatten(),
        torch::nn::Linear(256, 120),
        torch::nn::ReLU(),
        torch::nn::Linear(120, 84),
        torch::nn::ReLU(),
        torch::nn::Linear(84, 10));
    torch::load(net, "./checkpoint/model.pt");
    net->to(device);

    auto testset = torch::data::datasets::MNIST(
                       "../data/MNIST/raw",
                       torch::data::datasets::MNIST::Mode::kTest)
                       .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                       .map(torch::data::transforms::Stack<>());
    auto testloader = torch::data::make_data_loader<
        torch::data::samplers::SequentialSampler>(
        std::move(testset),
        torch::data::DataLoaderOptions().batch_size(64).workers(2));

    float test_loss = 0, test_acc = 0;
    int count_batch = 0, count_sample = 0;

    net->eval();
    torch::NoGradGuard no_grad;
    
    auto start = std::chrono::high_resolution_clock::now();
    for (torch::data::Example<>& batch : *testloader) {
        torch::Tensor images = batch.data.to(device);
        torch::Tensor labels = batch.target.to(device);

        torch::Tensor output = net->forward(images);
        torch::Tensor loss = torch::nn::functional::cross_entropy(output, labels);
        test_acc += torch::argmax(output, 1).eq(labels).sum().item<int64_t>();
        test_loss += loss.item<float>();

        count_batch += 1;
        count_sample += batch.data.size(0);
    }
    test_acc /= count_sample;
    test_loss /= count_batch;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    printf("test_loss: %.4f test_acc: %.4f time: %0.4f\n",
           test_loss, test_acc, diff.count());
    return 0;
}