#pragma once

#include <opencv2/opencv.hpp>

namespace ocv {

auto square_image(cv::Mat& image) -> cv::Mat {
    int64_t width = image.cols;
    int64_t height = image.rows;
    int64_t length = std::min(width, height);
    int64_t x = (width - length) / 2;
    int64_t y = (height - length) / 2;
    cv::Rect roi(x, y, length, length);
    return image(roi);
}

auto resize_image(cv::Mat& image, std::vector<int64_t> image_size) -> cv::Mat {
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(image_size[0], image_size[1]), 0, 0, cv::INTER_LINEAR);
    return resized_image;
}

auto load_image(const std::string& image_path) -> cv::Mat {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }
    return image;
}

}  // namespace ocv