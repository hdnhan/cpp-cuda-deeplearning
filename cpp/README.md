# Makefile
- Organize your project's files into directories, [reference](https://stackoverflow.com/questions/30573481/how-to-write-a-makefile-with-separate-source-and-header-directories).
- Multiple `main.cpp` files, [reference](http://www.lunderberg.com/2015/08/08/cpp-makefile/)

```bash
cpp/
|-- include/
|   |-- *.h      # header files
|-- src/
|   |-- *.cpp    # source files
|   |-- main.cpp # main file
|-- lib/         # third-party libraries
|-- bin/         # executables
|-- obj/         # object files
|-- Makefile
|-- make.sh      # script to run make
```

# CMakeList.txt
[CMake Tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)

```bash
cpp/
|-- include/
|   |-- *.h      # header files
|-- src/
|   |-- *.cpp    # source files
|   |-- main.cpp # main file
|-- lib/         # third-party libraries
|-- build/       # build files
|-- CMakeList.txt
|-- cmake.sh     # script to run cmake
```

# Third-party libraries
```bash
libMyLib/
|-- include/
|   |-- *.h      # header files
|-- lib/
|   |-- *.a      # static libraries
|   |-- *.so     # shared libraries
```