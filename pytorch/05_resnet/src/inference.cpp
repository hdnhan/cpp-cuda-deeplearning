#include <torch/torch.h>

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "cifar10.h"
#include "resnet.h"

int main() {
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << device << std::endl;

    // Define model
    const int64_t num_classes = 10;
    resnet::ResNet<resnet::BasicBlock> model = resnet::resnet18(num_classes);
    // resnet::ResNet<resnet::BasicBlock> model = resnet::resnet34(num_classes);
    // resnet::ResNet<resnet::Bottleneck> model = resnet::resnet50(num_classes);
    // resnet::ResNet<resnet::Bottleneck> model = resnet::resnet101(num_classes);
    // resnet::ResNet<resnet::Bottleneck> model = resnet::resnet152(num_classes);
    // resnet::ResNet<resnet::Bottleneck> model = resnet::resnext50_32x4d(num_classes);
    // resnet::ResNet<resnet::Bottleneck> model = resnet::resnext101_32x8d(num_classes);
    // resnet::ResNet<resnet::Bottleneck> model = resnet::wide_resnet50_2(num_classes);
    // resnet::ResNet<resnet::Bottleneck> model = resnet::wide_resnet101_2(num_classes);
    torch::load(model, "./checkpoint/model_cpp.pt");
    model->to(device);

    // Load the data
    const std::string CIFAR_data_path = "../data/cifar10/";
    auto testset = CIFAR10(CIFAR_data_path, CIFAR10::Mode::kTest)
                       .map(torch::data::transforms::Stack<>());
    auto testloader = torch::data::make_data_loader<
        torch::data::samplers::SequentialSampler>(
        std::move(testset),
        torch::data::DataLoaderOptions().batch_size(64).workers(2));

    float test_loss = 0, test_acc = 0;
    int count_batch = 0, count_sample = 0;

    model->eval();
    torch::NoGradGuard no_grad;

    auto start = std::chrono::high_resolution_clock::now();
    for (torch::data::Example<>& batch : *testloader) {
        torch::Tensor images = batch.data.to(device);
        torch::Tensor labels = batch.target.to(device);

        torch::Tensor output = model->forward(images);
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