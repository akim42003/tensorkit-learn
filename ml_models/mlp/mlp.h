#ifndef MLP_H
#define MLP_H

#include <vector>
#include <memory>
#include <string>
#include <cmath>
#include <random>
#include "../../tensor_cpp/tensor.h"

// Base class for activation functions
class Activation {
public:
    virtual ~Activation() = default;
    virtual Tensor forward(const Tensor& input) const = 0;
    virtual Tensor backward(const Tensor& input, const Tensor& grad_output) const = 0;
    virtual std::string name() const = 0;
};

// ReLU activation
class ReLU : public Activation {
public:
    Tensor forward(const Tensor& input) const override;
    Tensor backward(const Tensor& input, const Tensor& grad_output) const override;
    std::string name() const override { return "ReLU"; }
};

// Sigmoid activation
class Sigmoid : public Activation {
public:
    Tensor forward(const Tensor& input) const override;
    Tensor backward(const Tensor& input, const Tensor& grad_output) const override;
    std::string name() const override { return "Sigmoid"; }
};

// Tanh activation
class Tanh : public Activation {
public:
    Tensor forward(const Tensor& input) const override;
    Tensor backward(const Tensor& input, const Tensor& grad_output) const override;
    std::string name() const override { return "Tanh"; }
};

// Linear (identity) activation
class Linear : public Activation {
public:
    Tensor forward(const Tensor& input) const override;
    Tensor backward(const Tensor& input, const Tensor& grad_output) const override;
    std::string name() const override { return "Linear"; }
};

// Dense/Linear layer
class Layer {
private:
    int input_dim;
    int output_dim;
    std::shared_ptr<Activation> activation;
    Tensor weights;
    Tensor bias;
    
    // Cache for backpropagation
    Tensor input_cache;
    Tensor linear_output_cache;
    Tensor output_cache;
    
    // Weight initialization
    void initialize_weights(const std::string& init_method);
    
public:
    Layer(int input_dim, int output_dim, 
          std::shared_ptr<Activation> activation = std::make_shared<Linear>(),
          const std::string& init_method = "xavier");
    
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output, double learning_rate);
    
    // Getters
    Tensor get_weights() const { return weights; }
    Tensor get_bias() const { return bias; }
    int get_input_dim() const { return input_dim; }
    int get_output_dim() const { return output_dim; }
};

// Multi-Layer Perceptron
class MLP {
private:
    std::vector<std::shared_ptr<Layer>> layers;
    
public:
    MLP() = default;
    
    // Add a layer to the network
    void add_layer(int input_dim, int output_dim, 
                   std::shared_ptr<Activation> activation = std::make_shared<ReLU>(),
                   const std::string& init_method = "xavier");
    
    // Forward pass through entire network
    Tensor forward(const Tensor& input);
    
    // Backward pass through entire network
    void backward(const Tensor& loss_grad, double learning_rate);
    
    // Get number of layers
    int num_layers() const { return layers.size(); }
    
    // Get a specific layer
    std::shared_ptr<Layer> get_layer(int index) const { 
        return (index >= 0 && index < layers.size()) ? layers[index] : nullptr; 
    }
};

#endif // MLP_H