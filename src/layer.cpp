#include "layer.h"
#include <random>

LinearLayer::LinearLayer(int in_features, int out_features) {
    initialize(in_features, out_features);
}

void LinearLayer::initialize(int in_features, int out_features) {
    
    auto weights = std::make_shared<Tensor>(std::vector<int>{in_features, out_features});
    
    // Xavier/Glorot Initialization
    std::default_random_engine generator;
    double limit = std::sqrt(6.0 / (in_features + out_features));
    std::uniform_real_distribution<double> distribution(-limit, limit);
    
    for (int i = 0; i < weights->size; i++) {
        weights->data[i] = distribution(generator);
    }
    params.push_back(weights);

    auto bias = std::make_shared<Tensor>(std::vector<int>{1, out_features});
    for (int i = 0; i < bias->size; i++) {
        bias->data[i] = 0.0; // Standard to initialize bias to zero
    }
    params.push_back(bias);
}

std::shared_ptr<Tensor> LinearLayer::forward(std::shared_ptr<Tensor> input) {
    // Y = X * W + B
    return input->matmul(params[0])->add(params[1]);
}

std::shared_ptr<Tensor> ReLULayer::forward(std::shared_ptr<Tensor> input) {
    return input->relu();
}

std::shared_ptr<Tensor> FlattenLayer::forward(std::shared_ptr<Tensor> input) {
    return input->flatten();
}

Conv2DLayer::Conv2DLayer (int in_channels, int out_channels,int kernel_size,int stride, int padding){
    initialize(in_channels, out_channels,kernel_size,stride, padding);
}

void Conv2DLayer::initialize(int in_channels, int out_channels,int kernel_size,int stride, int padding){
    this->stride=stride;
    this->padding=padding;
    this->kernel_size=kernel_size;

    auto weights=std::make_shared<Tensor> (std::vector<int>{out_channels,in_channels,kernel_size,kernel_size});

    //He Initializtion
    std::default_random_engine generator;
    double limit = std::sqrt(2.0 / (in_channels*kernel_size*kernel_size));
    std::uniform_real_distribution<double> distribution(-limit, limit);
    
    for (int i = 0; i < weights->size; i++) {
        weights->data[i] = distribution(generator);
    }
    params.push_back(weights);

    auto bias = std::make_shared<Tensor>(std::vector<int>{1, out_channels});
    for (int i = 0; i < bias->size; i++) {
        bias->data[i] = 0.0; // Standard to initialize bias to zero
    }
    params.push_back(bias);
}

std::shared_ptr<Tensor> Conv2DLayer::forward(std::shared_ptr<Tensor> input) {
    return input->convolution(params[0],this->stride,this->padding,this->kernel_size)->add(params[1]);
}