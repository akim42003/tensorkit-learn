#ifndef LINK_FUNCTIONS_H
#define LINK_FUNCTIONS_H

#include "../../tensor_cpp/tensor.h"

class LinkFunction {
public:
    // Apply the link function: g(mu) = eta
    virtual Tensor operator()(const Tensor& mu) const = 0;

    // Apply the inverse link function: g^-1(eta) = mu
    virtual Tensor inverse(const Tensor& eta) const = 0;

    virtual ~LinkFunction() = default;
};

class IdentityLink : public LinkFunction {
public:
    Tensor operator()(const Tensor& mu) const override;
    Tensor inverse(const Tensor& eta) const override;
};

class LogitLink : public LinkFunction {
public:
    Tensor operator()(const Tensor& mu) const override;
    Tensor inverse(const Tensor& eta) const override;
};

class LogLink : public LinkFunction {
public:
    Tensor operator()(const Tensor& mu) const override;
    Tensor inverse(const Tensor& eta) const override;
};


#endif // LINK_FUNCTIONS_H
