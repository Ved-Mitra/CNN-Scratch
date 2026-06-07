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
    void conv2d(int in_channels, int out_channels, int kernel_size, int stride=1, int padding=0);
    void relu();
    void flatten();
    void maxpool2d(int pool_size,int stride=1,int padding=0);
    //train
    double train (std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> target);
    void fit(std::vector<std::shared_ptr<Tensor>> input,std::vector<std::shared_ptr<Tensor>> target, int epoch=10, int num_samples=100);
    std::vector<std::shared_ptr<Tensor>> test(std::vector<std::shared_ptr<Tensor>> input);
    double accuracy_score(std::vector<std::shared_ptr<Tensor>> pred,std::vector<std::shared_ptr<Tensor>> labels);

private:
    int epoch;
    double learning_rate;
    std::vector<std::shared_ptr<Layer>> layers;
    std::shared_ptr<Tensor> forward (std::shared_ptr<Tensor> input);
};