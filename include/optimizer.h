#include "tensor.h"
#include "layer.h"
#include "loss.h"

class SGD { //Stochastic Gradient Descent
public:
    SGD (std::vector<std::shared_ptr<Tensor>> params, double learning_rate);

    void step();
    void zero_grad();

private:
    std::vector<std::shared_ptr<Tensor>> params;
    double learning_rate;
};