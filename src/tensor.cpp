#include "tensor.h"

using namespace std;

Tensor::Tensor(vector<int> shape, Device device){

    //assigning class variables
    this->shape=shape;
    this->device=device;

    //calculating size of Tensor
    this->size=1;
    for(auto dim:shape){
        size*=dim;
    }

    //allocating memory based on device
    if(this->device==Device::CPU){
        this->data= new double[this->size]();
        this->grad= new double[this->size]();
    }
    else{
        // TODO
        cout << "CUDA to be implemented" << '\n';
    }
};

Tensor::~Tensor(){

    //deallocating memory
    if(this->device==Device::CPU){
        delete[] this->data;
        delete[] this->grad;
    }
};

int Tensor::get_index(const std::vector<int>& coords) const {
    int index = 0;
    int multiplier = 1;
    for (int i = shape.size() - 1; i >= 0; i--) {
        index += coords[i] * multiplier;
        multiplier *= shape[i];
    }
    return index;
}

shared_ptr<Tensor> Tensor::add(shared_ptr<Tensor> other){
    if(this->shape==other->shape) {   
        //standard addition
        auto out=make_shared<Tensor>(this->shape,this->device);
        for(int i=0;i<this->size;i++)
        {
            out->data[i]=this->data[i] + other->data[i];
        }

        //setting for bacward analysis
        out->children.push_back(shared_from_this());
        out->children.push_back(other);

        auto self=shared_from_this();
        out->backward_op=[self,other,out](){
            for(int i=0;i<out->size;i++){
                self->grad[i]+= out->grad[i]; // out->grad[i]*1
                other->grad[i]+= out->grad[i]; // out->grad[i]*1
            }
        };
        return out;
    }
    
    if(this->shape.size() == 3 && other->shape.size() == 2 && 
    this->shape[0] == other->shape[1] && other->shape[0] == 1) {
        //for convolution [O,H,W] + [1,O]
        auto out = std::make_shared<Tensor> (this->shape,this->device);
        int O=this->shape[0];
        int H=this->shape[1];
        int W=this->shape[2];
        for(int i=0;i<O;i++) {
            for(int j=0;j < H*W ;j++) {
                int idx = i*O + j;
                out->data[idx] = this->data[idx] + other->data[i];
            }
        }

        out->children.push_back(shared_from_this());
        out->children.push_back(other);

        auto self=shared_from_this();
        out->backward_op = [self,out,other,O,H,W] {
            for(int i=0;i<O;i++) {
                for(int j=0;j<H*W;j++) {
                    int idx=i*O + j;
                    self->grad[idx]+=out->grad[idx];
                    other->grad[i]+=out->grad[idx];
                }
            }
        };

        return out;
    }

    throw std::invalid_argument("Invalid Dimensions");
};

void Tensor::zero_grad(){
    for(int i=0;i<this->size;i++)
        this->grad[i]=0.0;
};

shared_ptr<Tensor> Tensor::mul(shared_ptr<Tensor> other){

    if(other->size!=this->size)
    {
        throw std::invalid_argument("Tensors are not of the same size");
    }

    //calculating new tensor
    auto out=make_shared<Tensor>(this->shape,this->device);
    for(int i=0;i<this->size;i++)
    {
        out->data[i]=this->data[i]*other->data[i];
    }

    //setting for bacward analysis
    out->children.push_back(shared_from_this());
    out->children.push_back(other);

    auto self=shared_from_this();
    out->backward_op=[self,other,out](){
        for(int i=0;i<out->size;i++){
            self->grad[i]+= out->grad[i]*other->data[i];
            other->grad[i]+= out->grad[i]*self->data[i];
        }
    };

    return out;
};

shared_ptr<Tensor> Tensor::matmul(shared_ptr<Tensor> other){
    if (this->shape.size() != 2 || other->shape.size() != 2) {
        throw std::invalid_argument("Matmul only supports 2D tensors for now.");
    }
    if(this->shape[1]!=other->shape[0]){
        throw std::invalid_argument("Tensors are not matrix multiplyable");
    }

    /*TODO --> optimize matrix multiplication using strassen/cache-optimized algorithm */
    int r1=this->shape[0],c1=this->shape[1],r2=other->shape[0],c2=other->shape[1];
    vector<int> newShape={r1,c2};
    auto out=make_shared<Tensor>(newShape,this->device);
    
    for(int i=0;i<r1;i++)
    {
        for(int j=0;j<c2;j++)
        {
            for(int k=0;k<c1;k++)
            {
                out->data[i*c2 + j]+= this->data[i*c1 + k]*other->data[k*c2 + j];
            }
        }
    }

    auto self=shared_from_this();
    out->backward_op=[self,other,out]{
        int rA=self->shape[0],cA=self->shape[1],rB=other->shape[0],cB=other->shape[1];
        int rC=rA,cC=cB;

        //self-grad
        for(int i=0;i<rA;i++)
        {
            for(int j=0;j<cA;j++)
            {
                for(int k=0;k<cB;k++)
                {
                    self->grad[i*cA + j]+= out->grad[i*cC + k]*other->data[k*cB + j];
                }
            }
        }

        //other-grad
        for(int i=0;i<rB;i++)
        {
            for(int j=0;j<cB;j++)
            {
                for(int k=0;k<rA;k++)
                {
                    other->grad[i*cB + j]+= self->data[k*cA + i]*out->grad[k*cC + j];
                }
            }
        }
    };

    return out;
}

void Tensor::backward(){
    set<shared_ptr<Tensor>> visited;
    vector<shared_ptr<Tensor>> topo;
    auto node=shared_from_this();
    dfs(visited,topo,node);

    // Seed the gradient of the loss tensor
    this->grad[0] = 1.0;

    // Execute backward ops in reverse topological order
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->backward_op) {
            (*it)->backward_op();
        }
    }
}

void Tensor::dfs(set<shared_ptr<Tensor>> &visited, vector<shared_ptr<Tensor>> &topo,shared_ptr<Tensor> &node){
    visited.insert(node);
    for(auto const &child : node->children){
        if(visited.find(child) == visited.end()){
            dfs(visited, topo, const_cast<shared_ptr<Tensor>&>(child));
        }
    }
    topo.push_back(node);
}

shared_ptr<Tensor> Tensor::relu(){
    auto out=make_shared<Tensor>(this->shape,this->device);
    for(int i=0;i<this->size;i++)
    {
        out->data[i]=max(0.0,this->data[i]);
    }
    
    out->children.push_back(shared_from_this());
    auto self = shared_from_this();
    
    out->backward_op = [self, out]() {
        for(int i = 0; i < out->size; i++) {
            self->grad[i] += (self->data[i] > 0) ? out->grad[i] : 0.0;
        }
    };
    
    return out;
}

shared_ptr<Tensor> Tensor::flatten() {
    auto out = std::make_shared<Tensor>(std::vector<int>{1, this->size}, this->device);
    // Copy data
    for(int i=0; i<this->size; i++) {
        out->data[i] = this->data[i];
    }
    
    // Autograd setup (Identity mapping)
    out->children.push_back(shared_from_this());
    auto self = shared_from_this();
    out->backward_op = [self, out]() {
        for(int i=0; i<self->size; i++) {
            self->grad[i] += out->grad[i];
        }
    };
    return out;
}

std::shared_ptr<Tensor> Tensor::convolution(std::shared_ptr<Tensor> other, int stride, int padding, int kernel_size) {
    // Input shape: [in_channels, in_height, in_width]
    // Filter shape: [out_channels, in_channels, k, k]
    int in_channels = this->shape[0];
    int in_height = this->shape[1];
    int in_width = this->shape[2];
    
    int out_channels = other->shape[0];
    int out_height = ((in_height + 2 * padding - kernel_size) / stride) + 1;
    int out_width = ((in_width + 2 * padding - kernel_size) / stride) + 1;

    auto out = std::make_shared<Tensor>(std::vector<int>{out_channels, out_height, out_width}, this->device);

    // Forward Pass
    for (int oc = 0; oc < out_channels; oc++) {
        for (int oh = 0; oh < out_height; oh++) {
            for (int ow = 0; ow < out_width; ow++) {
                double sum = 0.0;
                for (int ic = 0; ic < in_channels; ic++) {
                    for (int kh = 0; kh < kernel_size; kh++) {
                        for (int kw = 0; kw < kernel_size; kw++) {
                            int ih = oh * stride + kh - padding;
                            int iw = ow * stride + kw - padding;

                            if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                int input_idx = this->get_index({ic, ih, iw});
                                int kernel_idx = other->get_index({oc, ic, kh, kw});
                                sum += this->data[input_idx] * other->data[kernel_idx];
                            }
                        }
                    }
                }
                out->data[out->get_index({oc, oh, ow})] = sum;
            }
        }
    }

    out->children.push_back(shared_from_this());
    out->children.push_back(other);
    
    auto self = shared_from_this();
    out->backward_op = [self, out, other, stride, padding, kernel_size]() {
        int in_channels = self->shape[0];
        int in_height = self->shape[1];
        int in_width = self->shape[2];
        
        int out_channels = out->shape[0];
        int out_height = out->shape[1];
        int out_width = out->shape[2];

        for (int oc = 0; oc < out_channels; oc++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    int out_idx = out->get_index({oc, oh, ow});
                    double grad_val = out->grad[out_idx];

                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int kh = 0; kh < kernel_size; kh++) {
                            for (int kw = 0; kw < kernel_size; kw++) {
                                int ih = oh * stride + kh - padding;
                                int iw = ow * stride + kw - padding;

                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                    int input_idx = self->get_index({ic, ih, iw});
                                    int kernel_idx = other->get_index({oc, ic, kh, kw});

                                    // Gradient wrt weights
                                    other->grad[kernel_idx] += grad_val * self->data[input_idx];
                                    
                                    // Gradient wrt input
                                    self->grad[input_idx] += grad_val * other->data[kernel_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    };
    return out;
}