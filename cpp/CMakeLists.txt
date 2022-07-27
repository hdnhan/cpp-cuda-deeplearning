cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(CPP LANGUAGES CXX)

function(print)
    message("MY MESSAGE!!!")
    foreach(var ${ARGV})
        message("${var}: ${${var}}")
    endforeach()
endfunction()

print(CMAKE_VERSION PROJECT_NAME CMAKE_SOURCE_DIR)

file(GLOB SRC_FILES "src/*.cpp")
message("SRC_FILES: ${SRC_FILES}")
file(GLOB LIB_FILES "lib/lib/*.so")
message("LIB_FILES: ${LIB_FILES}")

# By default put binaries in build/bin (pre-install)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(main ${SRC_FILES})
target_include_directories(main PUBLIC include lib/include)
target_link_libraries(main ${LIB_FILES})
