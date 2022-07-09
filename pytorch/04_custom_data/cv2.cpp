#include <filesystem>
#include <iostream>
#include <string>
// #include <opencv2/opencv.hpp>

// using namespace std;
// using namespace cv;

// bash: g++ cv2.cpp -o out -std=c++17 `pkg-config --cflags --libs opencv4`
int main() {
    /*
    // Read image
    Mat img = imread("/home/nhan/Desktop/cpp-cuda-deeplearning/pytorch/data/pets/train/dog/dog.3383.jpg");
    cout << "Width : " << img.size().width << " " << img.cols << endl;
    cout << "Height: " << img.size().height << " " << img.rows << endl;
    cout << "Channels: :" << img.channels() << endl;
    // // Crop image
    img = img(Range(80, 280), Range(10, 210));
    cout << "Width : " << img.size().width << endl;
    cout << "Height: " << img.size().height << endl;
    cout << "Channels: :" << img.channels() << endl;
    */

    std::string data_dir = "/home/nhan/Desktop/cpp-cuda-deeplearning/pytorch/data/pets";
    for (auto &p : std::filesystem::directory_iterator(data_dir)) {
        std::cout << p.path().filename().string() << std::endl;
        if(p.is_directory()) std::cout << "dir" << std::endl;
        if(p.is_regular_file()) std::cout << "file" << std::endl;
    }
    return 0;
}