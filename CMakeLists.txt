cmake_minimum_required(VERSION 3.28)
project(digit_recognizer)

set(CMAKE_CXX_STANDARD 23)

find_package(Torch REQUIRED)

# Set OpenCV installation directory
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/Cellar/opencv/4.9.0_9.reinstall")

# Find OpenCV package
find_package(OpenCV COMPONENTS core highgui imgproc imgcodecs REQUIRED)

add_executable(digit_recognizer main.cpp)

# Include OpenCV headers
target_include_directories(digit_recognizer PUBLIC ${OpenCV_INCLUDE_DIRS})


target_link_libraries(digit_recognizer
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBRARIES}
)