#include "common.h"

namespace common {

cv::Mat square_image(cv::Mat& image) {
    int64_t width = image.cols;
    int64_t height = image.rows;
    int64_t length = std::min(width, height);
    int64_t x = (width - length) / 2;
    int64_t y = (height - length) / 2;
    cv::Rect roi(x, y, length, length);
    return image(roi);
}

cv::Mat resize_image(cv::Mat& image, std::vector<int64_t>& size) {
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(size[0], size[1]), 0, 0, cv::INTER_LINEAR);
    return resized_image;
}

cv::Mat load_image(const std::string& path) {
    cv::Mat image = cv::imread(path);
    if (image.empty())
        throw std::runtime_error("Failed to load image: " + path);
    return image;
}

}  // namespace common