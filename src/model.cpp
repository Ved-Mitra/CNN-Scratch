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
    std::shared_ptr<Tensor> loss = loss_func.CrossEntropy(pred, target);
    
    optimizer.zero_grad();
    loss->backward();

    //optimizer
    optimizer.step();

    return loss->data[0];
}

void SimpleModel::fit(std::vector<std::shared_ptr<Tensor>> input,std::vector<std::shared_ptr<Tensor>> target, int epoch, int num_samples){
    if(input.size()<num_samples){
        num_samples=input.size();
    }

    for(int e=0;e<epoch;e++){
        double total_epoch_loss=0.0;
        for(int i=0;i<num_samples;i++){
            total_epoch_loss+=train(input[i],target[i]);
        }
        std::cout << "Epoch " << e + 1 << " - Average Loss: " << (total_epoch_loss / num_samples) << std::endl;
    }
}

std::vector<std::shared_ptr<Tensor>> SimpleModel::test(std::vector<std::shared_ptr<Tensor>> input){
    std::vector<std::shared_ptr<Tensor>> out;
    for(int i=0;i<input.size();i++){
        out.push_back(forward(input[i]));
    }
    return out;
}

double SimpleModel::accuracy_score(std::vector<std::shared_ptr<Tensor>> pred, std::vector<std::shared_ptr<Tensor>> labels) {
    if (pred.size() != labels.size() || pred.empty()) return 0.0;
    
    double correct = 0.0;
    for (size_t i = 0; i < pred.size(); i++) {
        int pred_max_idx = 0;
        int label_max_idx = 0;
        for (int j = 0; j < pred[i]->size; j++) {
            if (pred[i]->data[j] > pred[i]->data[pred_max_idx]) {
                pred_max_idx = j;
            }
            if (labels[i]->data[j] > labels[i]->data[label_max_idx]) {
                label_max_idx = j;
            }
        }
        if (pred_max_idx == label_max_idx) {
            correct++;
        }
    }
    return correct / pred.size();
}

std::vector<std::shared_ptr<Tensor>>  SimpleModel::get_parameters() {
    std::vector<std::shared_ptr<Tensor>> params;
    for(auto layer : layers) {
        for(auto tensor:layer->params)
            params.push_back(tensor);
    }
    return params;
}