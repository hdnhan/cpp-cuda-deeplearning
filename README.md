**To compile programs in folder cuda/ which combine of these repos: [Nvidia Website](https://developer.nvidia.com/gpu-accelerated-libraries), [Nvidia Github Examples](https://github.com/NVIDIA/CUDALibrarySamples) and, [CoffeeBeforeArch Github](https://github.com/CoffeeBeforeArch/cuda_programming).**

Another interesting topic: int32_t vs float32 both have the same size in memory, but their behavior is quite different, read [stackoverflow](https://stackoverflow.com/questions/4806944/what-is-the-difference-between-the-float-and-integer-data-type-when-the-size-is), [wiki](https://en.wikipedia.org/wiki/IEEE_754-1985#cite_note-3https://en.wikipedia.org/wiki/IEEE_754-1985#cite_note-3) and, [wiki](https://en.wikipedia.org/wiki/Single-precision_floating-point_format), which heavily affect accuracy.

# Install CUDA
## Install Nvidia driver
```bash
sudo apt update
sudo apt install cmake git
sudo ubuntu-drivers install
# if not found, sudo apt-get install ubuntu-drivers-common
# check if the drivers are installed
nvidia-smi
```

## Install CUDA Toolkit
This https://developer.nvidia.com/cuda-toolkit-archive for scripts to install CUDA Toolkit.

<span style="color:red"> **Note** </span>:
- All steps here are for installing cuda-11.3. Replace cuda-11.3 with cuda-xx.x of your choice.
- cuda-11.3 => gcc-10 vs g++-10. Replace 10 with respective version (https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version).

### Runfile (local)
- Download file
  ```bash
  wget https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda_11.3.1_465.19.01_linux.run
  ```
- Check gcc vs g++ version
  ```bash
  sudo apt install gcc-10 g++-10
  gcc-10 --version
  g++-10 --version
  ```
- If multiple versions, use alias for `gcc` and `g++`
  ```bash
  sudo ln -sf /usr/bin/gcc-10 /usr/bin/gcc
  sudo ln -sf /usr/bin/g++-10 /usr/bin/g++
  gcc --version
  g++ --version
  ```
- Check whether nvidia driver is installed (type `nvidia-smi`). If yes, uncheck driver in the list
  ```bash
  sudo sh cuda_11.3.1_465.19.01_linux.run
  ```

### Deb (local)
### Deb (network)
### Common
#### Check version
```bash
ls /usr/local | grep cuda # => different cuda versions
/usr/local/cuda-11.3/bin/nvcc --version
```

### Use alias
If multiple versions, to use `cuda` instead `cuda-11.3`. Reference https://stackoverflow.com/questions/45477133/how-to-change-cuda-version
```bash
sudo ln -sfT /usr/local/cuda-11.3 /usr/local/cuda
/usr/local/cuda/bin/nvcc --version
```
