#include <torch/torch.h>

#include <iomanip>
#include <iostream>

#include "cifar10.h"
#include "resnet.h"
#include "transform.h"

int main() {
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << device << std::endl;

    // Hyper parameters
    const int64_t num_classes = 10;
    const int epochs = 10;
    const double learning_rate = 0.001;
    const size_t learning_rate_decay_frequency = 8;  // number of epochs after which to decay the learning rate
    const double learning_rate_decay_factor = 1.0 / 3.0;

    const std::string CIFAR_data_path = "../data/cifar10/";

    // Define model
    resnet::ResNet<resnet::BasicBlock> model = resnet::resnet18(num_classes);
    // resnet::ResNet<resnet::BasicBlock> model = resnet::resnet34(num_classes);
    // resnet::ResNet<resnet::Bottleneck> model = resnet::resnet50(num_classes);
    // resnet::ResNet<resnet::Bottleneck> model = resnet::resnet101(num_classes);
    // resnet::ResNet<resnet::Bottleneck> model = resnet::resnet152(num_classes);
    // resnet::ResNet<resnet::Bottleneck> model = resnet::resnext50_32x4d(num_classes);
    // resnet::ResNet<resnet::Bottleneck> model = resnet::resnext101_32x8d(num_classes);
    // resnet::ResNet<resnet::Bottleneck> model = resnet::wide_resnet50_2(num_classes);
    // resnet::ResNet<resnet::Bottleneck> model = resnet::wide_resnet101_2(num_classes);
    model->to(device);

    // CIFAR10 custom dataset
    auto trainset = CIFAR10(CIFAR_data_path, CIFAR10::Mode::kTrain)
                        .map(transform::ConstantPad(4))
                        .map(transform::RandomHorizontalFlip())
                        .map(transform::RandomCrop({32, 32}))
                        .map(torch::data::transforms::Stack<>());
    auto testset = CIFAR10(CIFAR_data_path, CIFAR10::Mode::kTest)
                       .map(torch::data::transforms::Stack<>());

    // Data loader
    auto trainloader = torch::data::make_data_loader<
        torch::data::samplers::RandomSampler>(
        std::move(trainset),
        torch::data::DataLoaderOptions().batch_size(64).workers(2));
    auto testloader = torch::data::make_data_loader<
        torch::data::samplers::SequentialSampler>(
        std::move(testset),
        torch::data::DataLoaderOptions().batch_size(64).workers(2));

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));

    auto current_learning_rate = learning_rate;

    for (int e = 0; e < epochs; ++e) {
        auto start = std::chrono::high_resolution_clock::now();
        float train_loss = 0, train_acc = 0;
        int count_batch = 0, count_sample = 0;

        model->train();
        for (torch::data::Example<>& batch : *trainloader) {
            auto images = batch.data.to(device);
            auto labels = batch.target.to(device);

            auto output = model->forward(images);
            auto loss = torch::nn::functional::cross_entropy(output, labels);
            train_acc += torch::argmax(output, 1).eq(labels).sum().item<int64_t>();
            train_loss += loss.item<float>();

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            count_batch += 1;
            count_sample += batch.data.size(0);
        }
        train_acc /= count_sample;
        train_loss /= count_batch;

        // Save model
        torch::save(model, "./checkpoint/model_cpp.pt");

        // Decay learning rate
        if ((e + 1) % learning_rate_decay_frequency == 0) {
            current_learning_rate *= learning_rate_decay_factor;
            static_cast<torch::optim::AdamOptions&>(optimizer.param_groups().front().options()).lr(current_learning_rate);
        }

        float test_loss = 0, test_acc = 0;
        count_batch = 0, count_sample = 0;
        model->eval();
        {
            torch::NoGradGuard no_grad;
            // c10::InferenceMode guard(true);
            for (torch::data::Example<>& batch : *testloader) {
                torch::Tensor images = batch.data.to(device);  //.to(torch::kFloat);
                torch::Tensor labels = batch.target.to(device);

                torch::Tensor output = model->forward(images);
                torch::Tensor loss = torch::nn::functional::cross_entropy(output, labels);
                test_acc += torch::argmax(output, 1).eq(labels).sum().item<int64_t>();
                test_loss += loss.item<float>();

                count_batch += 1;
                count_sample += batch.data.size(0);
            }
        }

        test_acc /= count_sample;
        test_loss /= count_batch;

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;

        printf("Epoch %d/%d: train_loss: %.4f train_acc: %.4f test_loss: %.4f test_acc: %.4f time: %0.4f\n",
               e + 1, epochs, train_loss, train_acc, test_loss, test_acc, diff.count());
    }

    return 0;
}
