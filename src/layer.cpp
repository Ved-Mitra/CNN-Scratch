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
    auto out = std::make_shared<Tensor>(std::vector<int>{1, input->size}, input->device);
    // Copy data
    for(int i=0; i<input->size; i++) {
        out->data[i] = input->data[i];
    }
    
    // Autograd setup (Identity mapping)
    out->children.push_back(input);
    out->backward_op = [input, out]() {
        for(int i=0; i<input->size; i++) {
            input->grad[i] += out->grad[i];
        }
    };
    return out;
}