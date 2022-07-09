#pragma once
#include <torch/torch.h>

namespace resnet {

torch::nn::Conv2d create_conv2d(int64_t in_planes,
                                int64_t out_planes,
                                int64_t kerner_size,
                                int64_t stride = 1,
                                int64_t padding = 0,
                                bool bias = false,
                                int64_t groups = 1,
                                int64_t dilation = 1);

torch::nn::Conv2d conv3x3(int64_t in_planes,
                          int64_t out_planes,
                          int64_t stride = 1,
                          int64_t groups = 1,
                          int64_t dilation = 1);

torch::nn::Conv2d conv1x1(int64_t in_planes,
                          int64_t out_planes,
                          int64_t stride = 1);

class BasicBlockImpl : public torch::nn::Module {
   public:
    BasicBlockImpl(int64_t inplanes,
                   int64_t planes,
                   int64_t stride = 1,
                   torch::nn::Sequential downsample = nullptr,
                   int64_t groups = 1,
                   int64_t base_width = 64,
                   int64_t dilation = 1);
    torch::Tensor forward(torch::Tensor x);
    static const int64_t m_expansion;

   private:
    torch::nn::Conv2d m_conv1{nullptr};
    torch::nn::BatchNorm2d m_bn1{nullptr};
    torch::nn::ReLU m_relu{nullptr};
    torch::nn::Conv2d m_conv2{nullptr};
    torch::nn::BatchNorm2d m_bn2{nullptr};
    torch::nn::Sequential m_downsample{nullptr};
    int64_t m_stride;
};
TORCH_MODULE(BasicBlock);

class BottleneckImpl : public torch::nn::Module {
   public:
    BottleneckImpl(int64_t inplanes,
                   int64_t planes,
                   int64_t stride = 1,
                   torch::nn::Sequential downsample = nullptr,
                   int64_t groups = 1,
                   int64_t base_width = 64,
                   int64_t dilation = 1);
    torch::Tensor forward(torch::Tensor x);
    static const int64_t m_expansion;

   private:
    torch::nn::Conv2d m_conv1{nullptr};
    torch::nn::BatchNorm2d m_bn1{nullptr};
    torch::nn::ReLU m_relu{nullptr};
    torch::nn::Conv2d m_conv2{nullptr};
    torch::nn::BatchNorm2d m_bn2{nullptr};
    torch::nn::Conv2d m_conv3{nullptr};
    torch::nn::BatchNorm2d m_bn3{nullptr};
    torch::nn::Sequential m_downsample{nullptr};
    int64_t m_stride;
};
TORCH_MODULE(Bottleneck);

template <typename Block>
class ResNetImpl : public torch::nn::Module {
   public:
    explicit ResNetImpl(const std::vector<int64_t> layers,
                        int64_t num_classes = 1000,
                        bool zero_init_residual = false,
                        int64_t groups = 1,
                        int64_t width_per_group = 64,
                        std::vector<int64_t> replace_stride_with_dilation = {});

    torch::Tensor _forward_impl(torch::Tensor x);
    torch::Tensor forward(torch::Tensor x);

   private:
    int64_t m_inplanes = 64;
    int64_t m_dilation = 1;
    int64_t m_groups = 1;
    int64_t m_base_width = 64;

    torch::nn::Conv2d m_conv1{nullptr};
    torch::nn::BatchNorm2d m_bn1{nullptr};
    torch::nn::ReLU m_relu{nullptr};
    torch::nn::MaxPool2d m_maxpool{nullptr};
    torch::nn::Sequential m_layer1{nullptr}, m_layer2{nullptr}, m_layer3{nullptr}, m_layer4{nullptr};
    torch::nn::AdaptiveAvgPool2d m_avgpool{nullptr};
    torch::nn::Linear m_fc{nullptr};

    torch::nn::Sequential _make_layer(int64_t planes,
                                      int64_t blocks,
                                      int64_t stride = 1,
                                      bool dilate = false);
};

template <typename Block>
class ResNet : public torch::nn::ModuleHolder<ResNetImpl<Block>> {
   public:
    using torch::nn::ModuleHolder<ResNetImpl<Block>>::ModuleHolder;
};

template <class Block>
ResNet<Block> _resnet(const std::vector<int64_t>& layers, int64_t num_classes = 1000,
                      bool zero_init_residual = false, int64_t groups = 1,
                      int64_t width_per_group = 64,
                      const std::vector<int64_t>& replace_stride_with_dilation = {});

ResNet<BasicBlock> resnet18(int64_t num_classes = 1000, bool zero_init_residual = false,
                            int64_t groups = 1, int64_t width_per_group = 64,
                            std::vector<int64_t> replace_stride_with_dilation = {});

ResNet<BasicBlock> resnet34(int64_t num_classes = 1000, bool zero_init_residual = false,
                            int64_t groups = 1, int64_t width_per_group = 64,
                            std::vector<int64_t> replace_stride_with_dilation = {});

ResNet<Bottleneck> resnet50(int64_t num_classes = 1000, bool zero_init_residual = false,
                            int64_t groups = 1, int64_t width_per_group = 64,
                            std::vector<int64_t> replace_stride_with_dilation = {});

ResNet<Bottleneck> resnet101(int64_t num_classes = 1000, bool zero_init_residual = false,
                             int64_t groups = 1, int64_t width_per_group = 64,
                             std::vector<int64_t> replace_stride_with_dilation = {});

ResNet<Bottleneck> resnet152(int64_t num_classes = 1000, bool zero_init_residual = false,
                             int64_t groups = 1, int64_t width_per_group = 64,
                             std::vector<int64_t> replace_stride_with_dilation = {});

ResNet<Bottleneck> resnext50_32x4d(int64_t num_classes = 1000, bool zero_init_residual = false,
                                   int64_t groups = 1, int64_t width_per_group = 64,
                                   std::vector<int64_t> replace_stride_with_dilation = {});

ResNet<Bottleneck> resnext101_32x8d(int64_t num_classes = 1000, bool zero_init_residual = false,
                                    int64_t groups = 1, int64_t width_per_group = 64,
                                    std::vector<int64_t> replace_stride_with_dilation = {});

ResNet<Bottleneck> wide_resnet50_2(int64_t num_classes = 1000, bool zero_init_residual = false,
                                   int64_t groups = 1, int64_t width_per_group = 64,
                                   std::vector<int64_t> replace_stride_with_dilation = {});

ResNet<Bottleneck> wide_resnet101_2(int64_t num_classes = 1000, bool zero_init_residual = false,
                                    int64_t groups = 1, int64_t width_per_group = 64,
                                    std::vector<int64_t> replace_stride_with_dilation = {});

}  // namespace resnet