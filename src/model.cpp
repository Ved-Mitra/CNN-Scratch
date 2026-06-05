#include "model.h"

SimpleModel :: SimpleModel (double learning_rate, int epoch) {
    this->learning_rate=learning_rate;
    this->epoch=epoch;
}

void SimpleModel::linearlayer(int in_features,int out_features){
    auto layer = std::make_shared<LinearLayer> (in_features,out_features);
    layers.push_back(layer);
}

void SimpleModel::relu(){
    auto layer = std::make_shared<ReLULayer>();
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