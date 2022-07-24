#include <torch/torch.h>

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "data.h"
#include "network.h"

int main() {
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << device << std::endl;

    // Define model
    network::Net1 net;
    torch::load(net, "./checkpoint/model.pt");
    net->to(device);

    // Load the data
    std::string data_dir = "../data/pets";
    std::vector<int64_t> image_size = {256, 256};

    auto testset = data::CsvDataset(data_dir, image_size, data::CsvDataset::Mode::kTest)
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