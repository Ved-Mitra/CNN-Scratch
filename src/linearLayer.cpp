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