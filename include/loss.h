#include "tensor.h"

class Loss {
    std::shared_ptr<Tensor> MSE(std::shared_ptr<Tensor> pred,std::shared_ptr<Tensor> target);
    
};