cmake_minimum_required(VERSION 3.10)
project(cnn_scratch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(include)

add_executable(cnn_scratch 
    main.cpp 
    src/tensor.cpp 
    src/layer.cpp 
    src/loss.cpp 
    src/optimizer.cpp 
    src/model.cpp
    src/dataloader.cpp
)