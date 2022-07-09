#pragma once

#include <torch/torch.h>

#include <fstream>
#include <string>
#include <vector>

#include "common.h"

namespace dataset {

// Read csv file and return (image_path, label)
auto read_csv(const std::string& csv_path) -> std::vector<std::pair<std::string, int>> {
    std::fstream in(csv_path, std::ios::in);
    std::string line, name, label;
    std::vector<std::pair<std::string, int>> samples;

    getline(in, line);  // header
    while (getline(in, line)) {
        std::stringstream s(line);
        getline(s, name, ',');
        getline(s, label, ',');
        samples.emplace_back(std::make_pair(name, stoi(label)));
    }
    return samples;
}

class CsvDataet : public torch::data::Dataset<CsvDataet> {
   public:
    enum class Mode {
        kTrain,
        kTest
    };

   private:
    Mode mode_;
    std::string mode_dir_;
    std::vector<int64_t> image_size_;
    std::vector<std::pair<std::string, int>> samples_;

   public:
    explicit CsvDataet(const std::string& data_dir,
                       std::vector<int64_t> image_size,
                       Mode mode = Mode::kTrain) {
        mode_ = mode;
        mode_dir_ = data_dir + "/" + (mode == Mode::kTrain ? "train" : "test");
        image_size_ = image_size;
        samples_ = read_csv(mode_dir_ + ".csv");
    };

    torch::data::Example<> get(size_t index) override {
        const auto& [image_name, label] = samples_[index];

        // Load image with OpenCV.
        auto image = ocv::load_image(mode_dir_ + "/" + (label == 1 ? "dog/" : "cat/") + image_name);
        image = ocv::square_image(image);
        image = ocv::resize_image(image, image_size_);
        image /= 255.0;

        torch::Tensor image_tensor = torch::from_blob(image.data, {image.rows, image.cols, 3}, torch::kByte).clone();
        image_tensor = image_tensor.permute({2, 0, 1});  // convert to CxHxW

        return {image_tensor, torch::tensor(label)};
    };

    torch::optional<size_t> size() const override {
        return samples_.size();
    };
};
}  // namespace dataset