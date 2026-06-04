#include "model.h"

SimpleModel :: SimpleModel (double learning_rate=0.1,int epoch=10) {
    this->learning_rate=learning_rate;
    this->epoch=epoch;
}

void SimpleModel::linearlayer(int in_features,int out_features){
    auto layer = std::make_shared<LinearLayer> (in_features,out_features);
    layers.push_back(layer);
}

std::shared_ptr<Tensor> SimpleModel::forward (std::shared_ptr<Tensor> input){
    auto out = input;
    for(auto layer:layers){
        out=layer->forward(out);
    }
    return out;
}

void SimpleModel::train(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> target) {
    //optimizer
    std::vector<std::shared_ptr<Tensor>> params=get_parameters();
    SGD optimizer(params,this->learning_rate);
    Loss loss_func;

    for(int i=0;i<epoch;i++) {
        //forward
        std::shared_ptr<Tensor> pred = forward(input);
        
        //loss
        std::shared_ptr<Tensor> loss = loss_func.MSE(pred, target);
        
        optimizer.zero_grad();
        loss->backward();

        //optimizer
        optimizer.step();

        //result
        std::cout << "Epoch " << i+1 << " : " << loss->data[0] << std::endl;
    }
}

std::vector<std::shared_ptr<Tensor>>  SimpleModel::get_parameters() {
    std::vector<std::shared_ptr<Tensor>> params;
    for(auto layer : layers) {
        for(auto tensor:layer->params)
            params.push_back(tensor);
    }
    return params;
}