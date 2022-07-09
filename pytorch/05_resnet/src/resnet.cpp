#include "resnet.h"

namespace resnet {

torch::nn::Conv2d create_conv2d(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
                                int64_t stride, int64_t padding, bool bias,
                                int64_t groups, int64_t dilation) {
    return torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size)
            .stride(stride)
            .padding(padding)
            .bias(bias)
            .groups(groups)
            .dilation(dilation));
}

torch::nn::Conv2d conv3x3(int64_t in_planes,
                          int64_t out_planes,
                          int64_t stride,
                          int64_t groups,
                          int64_t dilation) {
    return create_conv2d(in_planes, out_planes, 3, stride,
                         dilation, false, groups, dilation);
}

torch::nn::Conv2d conv1x1(int64_t in_planes,
                          int64_t out_planes,
                          int64_t stride) {
    return create_conv2d(in_planes, out_planes, 1, stride);
}

const int64_t BasicBlockImpl::m_expansion = 1;

BasicBlockImpl::BasicBlockImpl(int64_t inplanes,
                               int64_t planes,
                               int64_t stride,
                               torch::nn::Sequential downsample,
                               int64_t groups,
                               int64_t base_width,
                               int64_t dilation) {
    if (groups != 1 || base_width != 64)
        throw std::invalid_argument{"BasicBlock only supports groups=1 and base_width=64"};

    if (dilation > 1)
        throw std::invalid_argument{"Dilation > 1 not supported in BasicBlock"};

    m_conv1 = register_module("conv1", conv3x3(inplanes, planes, stride));
    m_bn1 = register_module("bn1", torch::nn::BatchNorm2d(planes));
    m_relu = register_module("relu", torch::nn::ReLU(true));
    m_conv2 = register_module("conv2", conv3x3(planes, planes));
    m_bn2 = register_module("bn2", torch::nn::BatchNorm2d(planes));
    if (downsample)
        m_downsample = register_module("downsample", downsample);
    m_stride = stride;
}

torch::Tensor BasicBlockImpl::forward(torch::Tensor x) {
    torch::Tensor identity = x;

    torch::Tensor out = m_conv1->forward(x);
    out = m_bn1->forward(out);
    out = m_relu->forward(out);

    out = m_conv2->forward(out);
    out = m_bn2->forward(out);

    if (m_downsample)
        identity = m_downsample->forward(x);

    out += identity;
    out = m_relu->forward(out);

    return out;
}

const int64_t BottleneckImpl::m_expansion = 4;

BottleneckImpl::BottleneckImpl(int64_t inplanes,
                               int64_t planes,
                               int64_t stride,
                               torch::nn::Sequential downsample,
                               int64_t groups,
                               int64_t base_width,
                               int64_t dilation) {
    int64_t width = planes * (base_width / 64) * groups;
    m_conv1 = register_module("conv1", conv1x1(inplanes, width));
    m_bn1 = register_module("bn1", torch::nn::BatchNorm2d{width});
    m_relu = register_module("relu", torch::nn::ReLU{true});
    m_conv2 = register_module("conv2", conv3x3(width, width, stride, groups, dilation));
    m_bn2 = register_module("bn2", torch::nn::BatchNorm2d{width});
    m_conv3 = register_module("conv3", conv1x1(width, planes * m_expansion));
    m_bn3 = register_module("bn3", torch::nn::BatchNorm2d{planes * m_expansion});
    if (downsample)
        m_downsample = register_module("downsample", downsample);
    m_stride = stride;
}

torch::Tensor BottleneckImpl::forward(torch::Tensor x) {
    torch::Tensor identity = x;

    torch::Tensor out = m_conv1->forward(x);
    out = m_bn1->forward(out);
    out = m_relu->forward(out);

    out = m_conv2->forward(out);
    out = m_bn2->forward(out);
    out = m_relu->forward(out);

    out = m_conv3->forward(out);
    out = m_bn3->forward(out);

    if (m_downsample)
        identity = m_downsample->forward(x);

    out += identity;
    out = m_relu->forward(out);

    return out;
}

template <typename Block>
ResNetImpl<Block>::ResNetImpl(const std::vector<int64_t> layers,
                              int64_t num_classes,
                              bool zero_init_residual,
                              int64_t groups,
                              int64_t width_per_group,
                              std::vector<int64_t> replace_stride_with_dilation) {
    if (replace_stride_with_dilation.size() == 0) {
        // Each element in the tuple indicates if we should replace
        // the 2x2 stride with a dilated convolution instead.
        replace_stride_with_dilation = {false, false, false};
    }
    if (replace_stride_with_dilation.size() != 3) {
        throw std::invalid_argument{
            "replace_stride_with_dilation should be empty or have exactly three elements."};
    }

    m_groups = m_groups;
    m_base_width = width_per_group;

    m_conv1 = register_module("conv1", create_conv2d(3, m_inplanes, 7, 2, 3, false, 1, 1));
    m_bn1 = register_module("bn1", torch::nn::BatchNorm2d{m_inplanes});
    m_relu = register_module("relu", torch::nn::ReLU{true});
    m_maxpool = register_module("maxpool", torch::nn::MaxPool2d{torch::nn::MaxPool2dOptions({3, 3}).stride({2, 2}).padding({1, 1})});

    m_layer1 = register_module("layer1", _make_layer(64, layers[0]));
    m_layer2 = register_module("layer2", _make_layer(128, layers[1], 2, replace_stride_with_dilation[0]));
    m_layer3 = register_module("layer3", _make_layer(256, layers[2], 2, replace_stride_with_dilation[1]));
    m_layer4 = register_module("layer4", _make_layer(512, layers[3], 2, replace_stride_with_dilation[2]));

    m_avgpool = register_module("avgpool", torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
    m_fc = register_module("fc", torch::nn::Linear(512 * Block::Impl::m_expansion, num_classes));

    // auto all_modules = modules(false);
    // https://pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_module.html#_CPPv4NK5torch2nn6Module7modulesEb
    for (auto m : modules(false)) {
        if (m->name() == "torch::nn::Conv2dImpl") {
            torch::OrderedDict<std::string, torch::Tensor>
                named_parameters = m->named_parameters(false);
            torch::Tensor* ptr_w = named_parameters.find("weight");
            torch::nn::init::kaiming_normal_(*ptr_w, 0, torch::kFanOut,
                                             torch::kReLU);
        } else if ((m->name() == "torch::nn::BatchNormImpl") ||
                   (m->name() == "torch::nn::GroupNormImpl")) {
            torch::OrderedDict<std::string, torch::Tensor>
                named_parameters = m->named_parameters(false);
            torch::Tensor* ptr_w = named_parameters.find("weight");
            torch::nn::init::constant_(*ptr_w, 1.0);
            torch::Tensor* ptr_b = named_parameters.find("bias");
            torch::nn::init::constant_(*ptr_b, 0.0);
        }
    }

    if (zero_init_residual) {
        for (auto m : modules(false)) {
            if (m->name() == "Bottleneck") {
                torch::OrderedDict<std::string, torch::Tensor>
                    named_parameters =
                        m->named_modules()["bn3"]->named_parameters(false);
                torch::Tensor* ptr_w = named_parameters.find("weight");
                torch::nn::init::constant_(*ptr_w, 0.0);
            } else if (m->name() == "BasicBlock") {
                torch::OrderedDict<std::string, torch::Tensor>
                    named_parameters =
                        m->named_modules()["bn2"]->named_parameters(false);
                torch::Tensor* ptr_w = named_parameters.find("weight");
                torch::nn::init::constant_(*ptr_w, 0.0);
            }
        }
    }
}

template <typename Block>
torch::Tensor ResNetImpl<Block>::_forward_impl(torch::Tensor x) {
    x = m_conv1->forward(x);
    x = m_bn1->forward(x);
    x = m_relu->forward(x);
    x = m_maxpool->forward(x);

    x = m_layer1->forward(x);
    x = m_layer2->forward(x);
    x = m_layer3->forward(x);
    x = m_layer4->forward(x);

    x = m_avgpool->forward(x);
    x = torch::flatten(x, 1);
    x = m_fc->forward(x);

    return x;
}

template <typename Block>
torch::Tensor ResNetImpl<Block>::forward(torch::Tensor x) {
    return ResNetImpl<Block>::_forward_impl(x);
}

template <typename Block>
torch::nn::Sequential ResNetImpl<Block>::_make_layer(int64_t planes,
                                                     int64_t blocks,
                                                     int64_t stride,
                                                     bool dilate) {
    torch::nn::Sequential downsample = nullptr;
    int64_t previous_dilation = m_dilation;
    if (dilate) {
        m_dilation *= stride;
        stride = 1;
    }
    if ((stride != 1) || (m_inplanes != planes * Block::Impl::m_expansion)) {
        downsample = torch::nn::Sequential{
            conv1x1(m_inplanes, planes * Block::Impl::m_expansion, stride),
            torch::nn::BatchNorm2d(planes * Block::Impl::m_expansion)};
    }

    torch::nn::Sequential layers;

    layers->push_back(Block(m_inplanes, planes, stride, downsample,
                            m_groups, m_base_width, previous_dilation));
    m_inplanes = planes * Block::Impl::m_expansion;
    for (int64_t i = 0; i < blocks; i++) {
        layers->push_back(Block(m_inplanes, planes, 1, nullptr, m_groups, m_base_width, m_dilation));
    }

    return layers;
}

template <class Block>
ResNet<Block> _resnet(const std::vector<int64_t>& layers, int64_t num_classes,
                      bool zero_init_residual, int64_t groups,
                      int64_t width_per_group,
                      const std::vector<int64_t>& replace_stride_with_dilation) {
    ResNet<Block> model = ResNet<Block>(layers, num_classes, zero_init_residual,
                                        groups, width_per_group, replace_stride_with_dilation);
    return model;
}

ResNet<BasicBlock> resnet18(int64_t num_classes, bool zero_init_residual,
                            int64_t groups, int64_t width_per_group,
                            std::vector<int64_t> replace_stride_with_dilation) {
    const std::vector<int64_t> layers{2, 2, 2, 2};
    ResNet<BasicBlock> model = _resnet<BasicBlock>(layers, num_classes, zero_init_residual, groups,
                                                   width_per_group, replace_stride_with_dilation);
    return model;
}

ResNet<BasicBlock> resnet34(int64_t num_classes, bool zero_init_residual,
                            int64_t groups, int64_t width_per_group,
                            std::vector<int64_t> replace_stride_with_dilation) {
    const std::vector<int64_t> layers{3, 4, 6, 3};
    ResNet<BasicBlock> model = _resnet<BasicBlock>(layers, num_classes, zero_init_residual, groups,
                                                   width_per_group, replace_stride_with_dilation);
    return model;
}

ResNet<Bottleneck> resnet50(int64_t num_classes, bool zero_init_residual,
                            int64_t groups, int64_t width_per_group,
                            std::vector<int64_t> replace_stride_with_dilation) {
    const std::vector<int64_t> layers{3, 4, 6, 3};
    ResNet<Bottleneck> model = _resnet<Bottleneck>(layers, num_classes, zero_init_residual, groups,
                                                   width_per_group, replace_stride_with_dilation);
    return model;
}

ResNet<Bottleneck> resnet101(int64_t num_classes, bool zero_init_residual,
                             int64_t groups, int64_t width_per_group,
                             std::vector<int64_t> replace_stride_with_dilation) {
    const std::vector<int64_t> layers{3, 4, 23, 3};
    ResNet<Bottleneck> model = _resnet<Bottleneck>(layers, num_classes, zero_init_residual, groups,
                                                   width_per_group, replace_stride_with_dilation);
    return model;
}

ResNet<Bottleneck> resnet152(int64_t num_classes, bool zero_init_residual,
                             int64_t groups, int64_t width_per_group,
                             std::vector<int64_t> replace_stride_with_dilation) {
    const std::vector<int64_t> layers{3, 8, 36, 3};
    ResNet<Bottleneck> model = _resnet<Bottleneck>(layers, num_classes, zero_init_residual, groups,
                                                   width_per_group, replace_stride_with_dilation);
    return model;
}

ResNet<Bottleneck> resnext50_32x4d(int64_t num_classes, bool zero_init_residual,
                                   int64_t groups, int64_t width_per_group,
                                   std::vector<int64_t> replace_stride_with_dilation) {
    groups = 32;
    width_per_group = 4;
    const std::vector<int64_t> layers{3, 4, 6, 3};
    ResNet<Bottleneck> model = _resnet<Bottleneck>(layers, num_classes, zero_init_residual, groups,
                                                   width_per_group, replace_stride_with_dilation);
    return model;
}

ResNet<Bottleneck> resnext101_32x8d(int64_t num_classes, bool zero_init_residual,
                                    int64_t groups, int64_t width_per_group,
                                    std::vector<int64_t> replace_stride_with_dilation) {
    groups = 32;
    width_per_group = 8;
    const std::vector<int64_t> layers{3, 4, 23, 3};
    ResNet<Bottleneck> model = _resnet<Bottleneck>(layers, num_classes, zero_init_residual, groups,
                                                   width_per_group, replace_stride_with_dilation);
    return model;
}

ResNet<Bottleneck> wide_resnet50_2(int64_t num_classes, bool zero_init_residual,
                                   int64_t groups, int64_t width_per_group,
                                   std::vector<int64_t> replace_stride_with_dilation) {
    width_per_group = 64 * 2;
    const std::vector<int64_t> layers{3, 4, 6, 3};
    ResNet<Bottleneck> model = _resnet<Bottleneck>(layers, num_classes, zero_init_residual, groups,
                                                   width_per_group, replace_stride_with_dilation);
    return model;
}

ResNet<Bottleneck> wide_resnet101_2(int64_t num_classes, bool zero_init_residual,
                                    int64_t groups, int64_t width_per_group,
                                    std::vector<int64_t> replace_stride_with_dilation) {
    width_per_group = 64 * 2;
    const std::vector<int64_t> layers{3, 4, 23, 3};
    ResNet<Bottleneck> model = _resnet<Bottleneck>(layers, num_classes, zero_init_residual, groups,
                                                   width_per_group, replace_stride_with_dilation);
    return model;
}

template class ResNetImpl<BasicBlock>;
template class ResNetImpl<Bottleneck>;

}  // namespace resnet