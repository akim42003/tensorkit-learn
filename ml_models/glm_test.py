import tensor_slow as ts
from glm import GLM
from optimizers.loss_functions import MSELoss, Binary_CELoss
from link_functions import IdentityLink, LogitLink

# Create synthetic dataset
X = ts.Tensor.from_values([4, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])  # 4 samples, 2 features
y = ts.Tensor.from_values([4, 1], [1.0, 0.0, 1.0, 0.0])  # Binary targets

# Initialize GLM with LogitLink and Binary Cross Entropy Loss
link_function = LogitLink()
loss_function = Binary_CELoss()
model = GLM(num_features=2, link_function=link_function, loss_function=loss_function)

# Train the model
model.train(X, y, learning_rate=0.01, epochs=100)

# Print the trained coefficients
print("Trained coefficients:")
model.get_coefficients().print()
