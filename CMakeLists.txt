cmake_minimum_required(VERSION 3.23)
project(DUHO)
set(CMAKE_CXX_STANDARD 20)

## Target
add_executable(${PROJECT_NAME} main.cpp
        header/superpixel.h source/superpixel.cpp
        header/eigen_image.h
        header/utils.h
        header/algorithm.h source/algorithm.cpp)

## Project files
target_include_directories(${PROJECT_NAME}
        PRIVATE ${PROJECT_SOURCE_DIR}/header
        PRIVATE ${PROJECT_SOURCE_DIR}/third_party)

target_link_directories(${PROJECT_NAME}
        PRIVATE ${PROJECT_SOURCE_DIR}/third_party)


## Link libraries
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/eigen)

find_package(PNG REQUIRED)
target_include_directories(${PROJECT_NAME}
        PRIVATE ${PNG_INCLUDE_DIR}
        PRIVATE third_party/eigen)
target_link_libraries(${PROJECT_NAME}
        PRIVATE ${PNG_LIBRARY})