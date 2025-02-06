import tensor_slow as ts
from link_functions import IdentityLink, LogitLink, LogLink

# Create a tensor
mu = ts.Tensor.from_values([2, 2], [0.2, 0.8, 0.6, 0.4])

# IdentityLink
identity = IdentityLink()
print("IdentityLink:")
print(identity(mu).print())
print(identity.inverse(mu).print())

# LogitLink
logit = LogitLink()
eta = logit(mu)
print("LogitLink (Forward):")
eta.print()

mu_inverse = logit.inverse(eta)
print("LogitLink (Inverse):")
mu_inverse.print()

# LogLink
log = LogLink()
log_eta = log(mu)
print("LogLink (Forward):")
log_eta.print()

log_mu = log.inverse(log_eta)
print("LogLink (Inverse):")
log_mu.print()
