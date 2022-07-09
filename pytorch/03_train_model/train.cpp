#include <torch/torch.h>

#include <chrono>
#include <iostream>

/*
struct ConvNetImpl : torch::nn::Module {
    ConvNetImpl() : backbone(
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
                        torch::nn::Linear(84, 10)) {
        register_module("backbone", backbone);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = backbone->forward(x);
        return x;
    }
    torch::nn::Sequential backbone;
};
*/

/*
class ConvNetImpl : public torch::nn::Module {
   private:
    torch::nn::Sequential backbone{
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
        torch::nn::Linear(84, 10)};

   public:
    ConvNetImpl() {
        register_module("backbone", backbone);
    }
    torch::Tensor forward(torch::Tensor x) {
        x = backbone->forward(x);
        return x;
    }
};
*/

// TORCH_MODULE(ConvNet);

int main() {
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << device << std::endl;

    // Define model
    // ConvNet net;
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
    net->to(device);

    // Load the data
    auto trainset = torch::data::datasets::MNIST(
                        "../data/MNIST/raw",
                        torch::data::datasets::MNIST::Mode::kTrain)
                        .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                        .map(torch::data::transforms::Stack<>());
    auto trainloader = torch::data::make_data_loader<
        torch::data::samplers::RandomSampler>(
        std::move(trainset),
        torch::data::DataLoaderOptions().batch_size(64).workers(2));

    auto testset = torch::data::datasets::MNIST(
                       "../data/MNIST/raw",
                       torch::data::datasets::MNIST::Mode::kTest)
                       .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                       .map(torch::data::transforms::Stack<>());
    auto testloader = torch::data::make_data_loader<
        torch::data::samplers::SequentialSampler>(
        std::move(testset),
        torch::data::DataLoaderOptions().batch_size(64).workers(2));

    torch::optim::SGD optimizer(net->parameters(), 0.01);

    int epochs = 10;
    for (int e = 0; e < epochs; e++) {
        auto start = std::chrono::high_resolution_clock::now();

        float train_loss = 0, train_acc = 0;
        int count_batch = 0, count_sample = 0;
        net->train();
        for (torch::data::Example<>& batch : *trainloader) {
            torch::Tensor images = batch.data.to(device);
            torch::Tensor labels = batch.target.to(device);

            torch::Tensor output = net->forward(images);
            torch::Tensor loss = torch::nn::functional::cross_entropy(output, labels);
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
        torch::save(net, "./checkpoint/model.pt");

        float test_loss = 0, test_acc = 0;
        count_batch = 0, count_sample = 0;
        net->eval();
        torch::NoGradGuard no_grad;
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

        printf("Epoch %d/%d: train_loss: %.4f train_acc: %.4f test_loss: %.4f test_acc: %.4f time: %0.4f\n",
               e + 1, epochs, train_loss, train_acc, test_loss, test_acc, diff.count());
    }

    return 0;
}
