#include "layer.h"
#include "optimizer.h"
#include "loss.h"

class SimpleModel {
public:
    //constructor
    SimpleModel(double learning_rate=0.1,int epoch=10);
    //optimizer
    std::vector<std::shared_ptr<Tensor>> get_parameters();
    //layer
    void linearlayer(int in_features,int out_features);
    void relu();
    //train
    double train (std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> target);

private:
    int epoch;
    double learning_rate; // Changed to double
    std::vector<std::shared_ptr<Layer>> layers; // Changed back to Layer base class
    std::shared_ptr<Tensor> forward (std::shared_ptr<Tensor> input);
};