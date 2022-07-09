#pragma once

#include <torch/torch.h>

#include <filesystem>  // c++17
#include <string>
#include <unordered_map>
#include <vector>

#include "common.h"

namespace dataset {

std::vector<std::string> parse_classes(const std::string &data_dir) {
    std::vector<std::string> classes;
    std::string train_dir = data_dir + "/train";

    for (auto &p : std::filesystem::directory_iterator(train_dir)) {
        if (p.is_directory()) {
            classes.push_back(p.path().filename().string());
        }
    }

    std::sort(classes.begin(), classes.end());
    return classes;
}

std::unordered_map<std::string, int> create_class_to_index_map(const std::vector<std::string> &classes) {
    std::unordered_map<std::string, int> class_to_index;

    int index = 0;
    for (const auto &class_name : classes) {
        class_to_index[class_name] = index++;
    }

    return class_to_index;
}

std::vector<std::pair<std::string, int>> create_samples(
    const std::string &mode_dir,
    const std::unordered_map<std::string, int> &class_to_index) {
    std::vector<std::pair<std::string, int>> samples;

    for (const auto &[class_name, class_index] : class_to_index) {
        for (const auto &p : std::filesystem::directory_iterator(mode_dir + "/" + class_name)) {
            if (p.is_regular_file()) {
                samples.emplace_back(p.path().string(), class_index);
            }
        }
    }

    return samples;
}

class FolderDataset : public torch::data::datasets::Dataset<FolderDataset> {
   public:
    enum class Mode {
        kTrain,
        kTest
    };

   private:
    Mode mode_;
    std::string mode_dir_;
    std::vector<int64_t> image_size_;
    std::vector<std::string> classes_;
    std::unordered_map<std::string, int> class_to_index_;
    std::vector<std::pair<std::string, int>> samples_;

   public:
    explicit FolderDataset(const std::string &data_dir,
                           std::vector<int64_t> image_size,
                           Mode mode = Mode::kTrain) {
        mode_ = mode;
        mode_dir_ = data_dir + "/" + (mode == Mode::kTrain ? "train" : "test");
        image_size_ = image_size;
        classes_ = parse_classes(data_dir);
        class_to_index_ = create_class_to_index_map(classes_);
        samples_ = create_samples(mode_dir_, class_to_index_);
    };

    torch::data::Example<> get(size_t index) override {
        const auto &[image_path, class_index] = samples_[index];

        // Using OpenCV to load image and transform.
        auto image = ocv::load_image(image_path);
        image = ocv::square_image(image);
        image = ocv::resize_image(image, image_size_);
        image /= 255.0;

        torch::Tensor image_tensor = torch::from_blob(image.data, {image.rows, image.cols, 3}, torch::kByte).clone();
        image_tensor = image_tensor.permute({2, 0, 1});  // convert to CxHxW

        return {image_tensor, torch::tensor(class_index)};
    };

    torch::optional<size_t> size() const override {
        return samples_.size();
    };
};
}  // namespace dataset