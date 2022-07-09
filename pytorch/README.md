**This installation guide is followed by** [README](../README.md)
# Install cuDNN
## Tar file installation (choose version of your choice)
```
sudo apt-get install xz-utils # try bellow first, if fails, try this
tar -xvf cudnn-11.3-linux-x64-v8.2.1.32.tar.xz 
sudo cp cuda/include/cudnn*.h /usr/local/cuda-11.3/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-11.3/lib64
sudo chmod a+r /usr/local/cuda-11.3/include/cudnn*.h /usr/local/cuda-11.3/lib64/libcudnn*
```

## Common
### Use alias
```bash
sudo ln -sfT /usr/local/cuda-11.3 /usr/local/cuda
```

# Install OpenCV:
- <span style="color:red"> **Install from source** </span>: https://linuxize.com/post/how-to-install-opencv-on-ubuntu-18-04/ 
- After installation, check `/usr/local/share/opencv*`


# LibTorch C++
*libtorch can be downloaded from the official pytorch website*
```bash
sudo apt-get install unzip # try bellow first, if fails, try this
unzip /path/to/libtorch-cxx11-abi-shared-with-deps-1.12.0+cu113.zip -d /path/to/cpp-cuda-deeplearning/pytorch/  
```

**Note**: To run on CPU only
- Download libtorch for cpu (much more lightweight than cuda)
- (*not important*) Comment out `CUDA_TOOLKIT_ROOT_DIR` and `CMAKE_CUDA_COMPILER` in `CMakeLists.txt`

**Reference**: https://github.com/prabhuomkar/pytorch-cpp
## 01_hello_world
```
mkdir build
./run.sh
```

## 02_pretrained_model
- Training by using Python and Pytorch
  - Save the model state
  - Save the traced model
- Inference using Python or C++ (with Pytorch)
  - Load the model state by Python
  - Load the traced model by C++ (why input is IValue???)


## 03_train_model
- Training by using C++ with libtorch
  - Network is defined by torch::nn::Sequential, struct, class
  - Using predefined datasets such as torch::data::datasets::MNIST
  - torch::save()
- Inference using C++ with libtorch
  - torch::load()
  - can't load model using Python????

## 04_custom_data
- Training by using C++ with libtorch
  - Network is defined by struct, class
  - Using custom dataset with csv file
  - torch::save()
- Inference using C++ with libtorch
  - torch::load()

## 05_resnet
- Training by using C++ with libtorch
  - Network is defined by torch::nn::Sequential, struct, class
  - Using Cifar10 dataset
  - torch::save()