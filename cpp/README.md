# Makefile
Organize your project's files into directories, [reference](https://stackoverflow.com/questions/30573481/how-to-write-a-makefile-with-separate-source-and-header-directories).
```bash
cpp/
|-- include/
|   |-- *.h   # header files
|-- src/
|   |-- *.cpp # source files
|-- lib/      # third-party libraries
|   |-- *.a   # static libraries
|   |-- *.so  # shared libraries
|-- bin/      # executables
|-- obj/      # object files
|-- Makefile
|-- make.sh   # script to run make
```

# CMakeList.txt
