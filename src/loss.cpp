#include "loss.h"

std::shared_ptr<Tensor> Loss::MSE(std::shared_ptr<Tensor> pred,std::shared_ptr<Tensor> target) {
    if(pred->size!=target->size){
        throw std::invalid_argument("Prediction and Target don't have same size");
    }

    auto out = std::make_shared<Tensor> (std::vector<int>{1},pred->device);

    double sum_sq_diff=0.0;
    for(int i=0;i<pred->size;i++)
    {
        double diff = (pred->data[i]-target->data[i]);
        sum_sq_diff+=(diff*diff);
    }
    out->data[0]=(sum_sq_diff/pred->size);

    out->backward_op = [pred,target,out] {
        for(int i=0; i < pred->size; i++) {
            // Chain rule: pred_grad += (local_deriv) * out_grad
            double local_deriv = (2.0 * (pred->data[i] - target->data[i])) / pred->size;
            pred->grad[i] += local_deriv*out->grad[0];
        }
    };
}