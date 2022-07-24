#pragma once

#include <opencv2/opencv.hpp>

namespace common {

cv::Mat square_image(cv::Mat& image);
cv::Mat resize_image(cv::Mat& image, std::vector<int64_t>& size);
cv::Mat load_image(const std::string& path);

}  // namespace common