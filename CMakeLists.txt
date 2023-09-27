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

## Download submodules
find_package(GIT QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

## Check submodules
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/third_party/eigen/CMakeLists.txt")
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

## Link libraries
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/eigen)

find_package(PNG REQUIRED)
find_package(TBB REQUIRED COMPONENTS tbb)
target_include_directories(${PROJECT_NAME}
        PRIVATE ${PNG_INCLUDE_DIR}
        PRIVATE third_party/eigen)
target_link_libraries(${PROJECT_NAME}
        PRIVATE ${PNG_LIBRARY}
        PRIVATE tbb)