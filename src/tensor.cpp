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

    if(other->size!=this->size)
    {
        throw std::invalid_argument("Tensors are not of the same size");
    }

    //calculating new tensor
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