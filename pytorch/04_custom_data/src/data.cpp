#include "data.h"

namespace data {

std::vector<std::pair<std::string, int>> read_csv(const std::string &path) {
    std::fstream in(path, std::ios::in);
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

CsvDataset::CsvDataset(const std::string &data_dir,
                       std::vector<int64_t> image_size,
                       Mode mode) : mode_(mode), image_size_(image_size) {
    mode_dir_ = data_dir + "/" + (mode == Mode::kTrain ? "train" : "test");
    samples_ = read_csv(mode_dir_ + ".csv");
}

torch::data::Example<> CsvDataset::get(size_t index) {
    const auto &[name, label] = samples_[index];

    // Load image with OpenCV.
    auto image = common::load_image(mode_dir_ + "/" + (label == 1 ? "dog/" : "cat/") + name);
    image = common::square_image(image);
    image = common::resize_image(image, image_size_);
    image /= 255.0;

    torch::Tensor image_tensor = torch::from_blob(image.data, {image.rows, image.cols, 3}, torch::kByte).clone();
    image_tensor = image_tensor.permute({2, 0, 1});  // convert to CxHxW

    return {image_tensor, torch::tensor(label)};
};

torch::optional<size_t> CsvDataset::size() const {
    return samples_.size();
};

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

FolderDataset::FolderDataset(const std::string &data_dir,
                             std::vector<int64_t> image_size,
                             Mode mode) {
    mode_ = mode;
    mode_dir_ = data_dir + "/" + (mode == Mode::kTrain ? "train" : "test");
    image_size_ = image_size;
    classes_ = parse_classes(data_dir);
    class_to_index_ = create_class_to_index_map(classes_);
    samples_ = create_samples(mode_dir_, class_to_index_);
};

torch::data::Example<> FolderDataset::get(size_t index) {
    const auto &[image_path, class_index] = samples_[index];

    // Using OpenCV to load image and transform.
    auto image = common::load_image(image_path);
    image = common::square_image(image);
    image = common::resize_image(image, image_size_);
    image /= 255.0;

    torch::Tensor image_tensor = torch::from_blob(image.data, {image.rows, image.cols, 3}, torch::kByte).clone();
    image_tensor = image_tensor.permute({2, 0, 1});  // convert to CxHxW

    return {image_tensor, torch::tensor(class_index)};
};

torch::optional<size_t> FolderDataset::size() const {
    return samples_.size();
};

}  // namespace data