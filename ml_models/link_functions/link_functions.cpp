#include "link_functions.h"
#include <stdexcept>
#include <cmath>
#include <vector>
#include <numeric> // For std::accumulate

// Helper function to compute the total number of elements in a tensor
size_t computeTotalElements(const std::vector<size_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
}

// --- IdentityLink ---
Tensor IdentityLink::operator()(const Tensor& mu) const {
    return mu; // g(mu) = mu
}

Tensor IdentityLink::inverse(const Tensor& eta) const {
    return eta; // g^-1(eta) = eta
}

// --- LogitLink ---
Tensor LogitLink::operator()(const Tensor& mu) const {
    // Compute 1 - mu using a tensor of ones
    size_t total_elements = computeTotalElements(mu.getShape());
    Tensor one_minus_mu = Tensor::from_values(mu.getShape(), std::vector<float>(total_elements, 1.0)).tminus(mu);

    // Compute log(mu) - log(1 - mu)
    return mu.log().tminus(one_minus_mu.log());
}

Tensor LogitLink::inverse(const Tensor& eta) const {
    // Compute exp(-eta) and 1 + exp(-eta)
    Tensor exp_neg_eta = eta.tminus(Tensor::zeros(eta.getShape())).exp();

    // Create a tensor of ones with the same shape as eta
    size_t total_elements = computeTotalElements(eta.getShape());
    Tensor one = Tensor::from_values(eta.getShape(), std::vector<float>(total_elements, 1.0));

    // Compute 1 / (1 + exp(-eta))
    return exp_neg_eta.divide(exp_neg_eta.tplus(one));
}

// --- LogLink ---
Tensor LogLink::operator()(const Tensor& mu) const {
    return mu.log(); // g(mu) = log(mu)
}

Tensor LogLink::inverse(const Tensor& eta) const {
    return eta.exp(); // g^-1(eta) = exp(eta)
}


// int main() {
//     // Test IdentityLink
//     IdentityLink identity;
//     Tensor t = Tensor::from_values({2, 2}, {1.0, 2.0, 3.0, 4.0});
//     Tensor identity_result = identity(t);
//     identity_result.print();

//     // Test LogitLink
//     LogitLink logit;
//     Tensor probabilities = Tensor::from_values({2, 2}, {0.2, 0.8, 0.6, 0.4});
//     Tensor logit_result = logit(probabilities);
//     logit_result.print();

//     Tensor inverse_logit_result = logit.inverse(logit_result);
//     inverse_logit_result.print();

//     // Test LogLink
//     LogLink log;
//     Tensor positive_values = Tensor::from_values({2, 2}, {1.0, 2.0, 3.0, 4.0});
//     Tensor log_result = log(positive_values);
//     log_result.print();

//     Tensor inverse_log_result = log.inverse(log_result);
//     inverse_log_result.print();

//     return 0;
// }