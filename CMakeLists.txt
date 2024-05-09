cmake_minimum_required(VERSION 3.15)
project(athlete-spy CXX)

find_package(tensorflowlite REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(athlete-spy src/main.cpp src/utils/KeypointData.h src/utils/PoseData.h src/utils/Transformer.h src/utils/Model.h src/utils/Visualizer.h src/utils/KeypointData.cpp src/utils/PoseData.cpp src/utils/Transformer.cpp src/utils/Model.cpp)

target_link_libraries(athlete-spy PRIVATE tensorflow::tensorflowlite opencv::opencv)
