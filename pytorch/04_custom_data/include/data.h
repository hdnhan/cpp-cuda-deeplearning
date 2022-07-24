#pragma once

#include <torch/torch.h>

#include <filesystem>  // c++17
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "common.h"

namespace data {

/***
 * Reads a CSV file and returns a vector of pairs of image name and label.
 * @param path The path to the CSV file.
 * @return A vector of pairs of image name and label.
 */
std::vector<std::pair<std::string, int>> read_csv(const std::string &path);

class CsvDataset : public torch::data::Dataset<CsvDataset> {
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
    explicit CsvDataset(const std::string &data_dir,
                        std::vector<int64_t> image_size,
                        Mode mode = Mode::kTrain);
    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override;
};

std::vector<std::string> parse_classes(const std::string &data_dir);

std::unordered_map<std::string, int> create_class_to_index_map(const std::vector<std::string> &classes);

std::vector<std::pair<std::string, int>> create_samples(
    const std::string &mode_dir,
    const std::unordered_map<std::string, int> &class_to_index);

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
                           Mode mode = Mode::kTrain);
    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override;
};

}  // namespace data