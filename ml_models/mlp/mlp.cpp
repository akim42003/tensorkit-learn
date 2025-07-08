#include "mlp.h"
#include <algorithm>
#include <iostream>


Tensor add_bias(const Tensor& input, const Tensor& bias) {
    // input shape: [batch_size, features]
    // bias shape: [features]
    Tensor result = Tensor::zeros(input.getShape());
    size_t batch_size = input.getShape()[0];
    size_t features = input.getShape()[1];
    
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t f = 0; f < features; f++) {
            result({b, f}) = input({b, f}) + bias({f});
        }
    }
    
    return result;
}


// ReLU implementation
Tensor ReLU::forward(const Tensor& input) const {
    return input.clamp(0.0, std::numeric_limits<float>::infinity());
}

Tensor ReLU::backward(const Tensor& input, const Tensor& grad_output) const {
    // Derivative is 1 for x > 0, 0 otherwise
    std::vector<size_t> shape = input.getShape();
    size_t total_size = 1;
    for (size_t dim : shape) {
        total_size *= dim;
    }
    
    std::vector<float> mask_data;
    mask_data.reserve(total_size);
    
    // Create mask based on input values
    for (size_t i = 0; i < total_size; i++) {
        std::vector<size_t> indices(shape.size(), 0);
        size_t idx = i;
        for (int d = shape.size() - 1; d >= 0; d--) {
            indices[d] = idx % shape[d];
            idx /= shape[d];
        }
        mask_data.push_back(input(indices) > 0 ? 1.0f : 0.0f);
    }
    
    Tensor mask = Tensor::from_values(shape, mask_data);
    return grad_output.elementwise_multiply(mask);
}

// Sigmoid implementation
Tensor Sigmoid::forward(const Tensor& input) const {
    // sigmoid(x) = 1 / (1 + exp(-x))
    // Clamp input to avoid overflow
    Tensor x_clamped = input.clamp(-500.0, 500.0);
    Tensor ones = Tensor::ones(input.getShape());
    
    // Create -x using scalar multiplication
    Tensor neg_x_clamped = x_clamped.scalar_multiply(-1.0);
    
    Tensor exp_neg_x = neg_x_clamped.exp();
    Tensor denominator = ones.tplus(exp_neg_x);
    return ones.divide(denominator);
}

Tensor Sigmoid::backward(const Tensor& input, const Tensor& grad_output) const {
    // Derivative: sigmoid(x) * (1 - sigmoid(x))
    Tensor sigmoid_x = forward(input);
    Tensor ones = Tensor::ones(input.getShape());
    Tensor one_minus_sigmoid = ones.tminus(sigmoid_x);
    Tensor derivative = sigmoid_x.elementwise_multiply(one_minus_sigmoid);
    return grad_output.elementwise_multiply(derivative);
}

// Tanh implementation
Tensor Tanh::forward(const Tensor& input) const {
    // tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    Tensor x_clamped = input.clamp(-500.0, 500.0);
    Tensor exp_x = x_clamped.exp();
    
    // Create -x using scalar multiplication
    Tensor neg_x_clamped = x_clamped.scalar_multiply(-1.0);
    Tensor exp_neg_x = neg_x_clamped.exp();
    
    Tensor numerator = exp_x.tminus(exp_neg_x);
    Tensor denominator = exp_x.tplus(exp_neg_x);
    return numerator.divide(denominator);
}

Tensor Tanh::backward(const Tensor& input, const Tensor& grad_output) const {
    // Derivative: 1 - tanh(x)^2
    Tensor tanh_x = forward(input);
    Tensor tanh_x_squared = tanh_x.elementwise_multiply(tanh_x);
    Tensor ones = Tensor::ones(input.getShape());
    Tensor derivative = ones.tminus(tanh_x_squared);
    return grad_output.elementwise_multiply(derivative);
}

// Linear implementation
Tensor Linear::forward(const Tensor& input) const {
    return input;
}

Tensor Linear::backward(const Tensor& input, const Tensor& grad_output) const {
    return grad_output;
}

// Layer implementation
Layer::Layer(int input_dim, int output_dim, 
             std::shared_ptr<Activation> activation,
             const std::string& init_method)
    : input_dim(input_dim), output_dim(output_dim), activation(activation),
      weights({1}), bias({1}), input_cache({1}), linear_output_cache({1}), output_cache({1}) {
    initialize_weights(init_method);
}

void Layer::initialize_weights(const std::string& init_method) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    double std_dev;
    if (init_method == "xavier") {
        // Xavier/Glorot initialization: sqrt(2 / (fan_in + fan_out))
        std_dev = std::sqrt(2.0 / (input_dim + output_dim));
    } else if (init_method == "he") {
        // He initialization: sqrt(2 / fan_in)
        std_dev = std::sqrt(2.0 / input_dim);
    } else {
        std_dev = 0.01;
    }
    
    std::normal_distribution<double> dist(0.0, std_dev);
    
    // Initialize weights
    std::vector<float> weight_data;
    weight_data.reserve(input_dim * output_dim);
    for (int i = 0; i < input_dim * output_dim; ++i) {
        weight_data.push_back(static_cast<float>(dist(gen)));
    }
    weights = Tensor::from_values({static_cast<size_t>(input_dim), static_cast<size_t>(output_dim)}, weight_data);
    
    // Initialize bias to small values
    std::vector<float> bias_data(output_dim, 0.01f);
    bias = Tensor::from_values({static_cast<size_t>(output_dim)}, bias_data);
}

Tensor Layer::forward(const Tensor& input) {
    // Cache input for backpropagation
    input_cache = input;
    
    // Linear transformation: input @ weights + bias
    linear_output_cache = input.matmul(weights);
    // Add bias to each row
    linear_output_cache = add_bias(linear_output_cache, bias);
    
    // Apply activation
    output_cache = activation->forward(linear_output_cache);
    return output_cache;
}

Tensor Layer::backward(const Tensor& grad_output, double learning_rate) {
    // Gradient through activation function
    Tensor grad_activation = activation->backward(linear_output_cache, grad_output);
    
    // Gradient w.r.t weights: input.T @ grad_activation
    Tensor grad_weights = input_cache.Tp().matmul(grad_activation);
    
    // Gradient w.r.t bias: sum over batch dimension (first dimension)
    std::vector<float> grad_bias_data(output_dim, 0.0f);
    std::vector<size_t> grad_shape = grad_activation.getShape();
    int batch_size = grad_shape[0];
    
    for (int b = 0; b < batch_size; ++b) {
        for (int j = 0; j < output_dim; ++j) {
            grad_bias_data[j] += grad_activation({static_cast<size_t>(b), static_cast<size_t>(j)});
        }
    }
    Tensor grad_bias = Tensor::from_values({static_cast<size_t>(output_dim)}, grad_bias_data);
    
    // Gradient w.r.t input: grad_activation @ weights.T
    Tensor grad_input = grad_activation.matmul(weights.Tp());
    
    // Update weights and bias
    weights = weights.tminus(grad_weights.scalar_multiply(learning_rate));
    bias = bias.tminus(grad_bias.scalar_multiply(learning_rate));
    
    return grad_input;
}

// MLP implementation
void MLP::add_layer(int input_dim, int output_dim, 
                    std::shared_ptr<Activation> activation,
                    const std::string& init_method) {
    layers.push_back(std::make_shared<Layer>(input_dim, output_dim, activation, init_method));
}

Tensor MLP::forward(const Tensor& input) {
    Tensor output = input;
    for (auto& layer : layers) {
        output = layer->forward(output);
    }
    return output;
}

void MLP::backward(const Tensor& loss_grad, double learning_rate) {
    Tensor grad = loss_grad;
    // Propagate gradients backward through layers
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        grad = (*it)->backward(grad, learning_rate);
    }
}