#include "model.h"

SimpleModel :: SimpleModel (double learning_rate, int epoch) {
    this->learning_rate=learning_rate;
    this->epoch=epoch;
}

void SimpleModel::linearlayer(int in_features,int out_features){
    auto layer = std::make_shared<LinearLayer> (in_features,out_features);
    layers.push_back(layer);
}

void SimpleModel::conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding) {
    auto layer = std::make_shared<Conv2DLayer>(in_channels, out_channels, kernel_size, stride, padding);
    layers.push_back(layer);
}

void SimpleModel::relu(){
    auto layer = std::make_shared<ReLULayer>();
    layers.push_back(layer);
}

void SimpleModel::flatten(){
    auto layer = std::make_shared<FlattenLayer>();
    layers.push_back(layer);
}

void SimpleModel::maxpool2d(int pool_size, int stride, int padding) {
    auto layer= std::make_shared<Max2DPoolLayer>(pool_size,stride,padding);
    layers.push_back(layer);
}

std::shared_ptr<Tensor> SimpleModel::forward (std::shared_ptr<Tensor> input){
    auto out = input;
    for(auto layer:layers){
        out=layer->forward(out);
    }
    return out;
}

double SimpleModel::train(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> target) {
    //optimizer
    std::vector<std::shared_ptr<Tensor>> params=get_parameters();
    SGD optimizer(params,this->learning_rate);
    Loss loss_func;
    
    //forward
    std::shared_ptr<Tensor> pred = forward(input);
    
    //loss
    std::shared_ptr<Tensor> loss = loss_func.MSE(pred, target);
    
    optimizer.zero_grad();
    loss->backward();

    //optimizer
    optimizer.step();

    return loss->data[0];
}

std::vector<std::shared_ptr<Tensor>>  SimpleModel::get_parameters() {
    std::vector<std::shared_ptr<Tensor>> params;
    for(auto layer : layers) {
        for(auto tensor:layer->params)
            params.push_back(tensor);
    }
    return params;
}