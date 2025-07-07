# tensorkit-learn

A tensor-based machine learning library built from scratch for educational purposes. This project implements fundamental ML algorithms using a custom C++ tensor library with Python bindings, providing a deep understanding of how modern ML frameworks work under the hood.

## Overview

tensorkit-learn is a hybrid C++/Python machine learning framework that demonstrates the implementation of core ML concepts from first principles. It features a custom tensor library written in C++ for performance, with Python bindings for ease of use and high-level algorithm implementation.

### Key Features

- **Custom Tensor Library**: N-dimensional array operations implemented in C++ with full broadcasting support
- **Machine Learning Models**: GLM (Generalized Linear Models) and SVM (Support Vector Machines) with kernel methods
- **Modular Architecture**: Separate components for tensors, optimizers, loss functions, and models
- **Educational Focus**: Clean, readable code prioritizing understanding over optimization
- **Python Bindings**: Seamless integration between C++ performance and Python flexibility

## Architecture

The project follows a layered architecture:

```
┌─────────────────────────────────────────┐
│         Python ML Models                │
│      (GLM, SVM, Optimizers)            │
├─────────────────────────────────────────┤
│      Python Bindings (pybind11)         │
│         tensor_slow module              │
├─────────────────────────────────────────┤
│         C++ Core Libraries              │
│  (Tensor, Link Functions, DataLoader)   │
└─────────────────────────────────────────┘
```

## Project Structure

```
tensorkit-learn/
├── tensor_cpp/             # Core C++ tensor implementation
│   ├── tensor.h           # Tensor class definition
│   └── tensor.cpp         # Tensor operations implementation
├── bindings/              # Python-C++ bindings
│   └── bindings.cpp       # pybind11 module definition
├── ml_models/             # Machine learning algorithms
│   ├── glm.py            # Generalized Linear Models
│   ├── svm.py            # Support Vector Machine
│   ├── kernel_func.py    # Kernel functions for SVM
│   ├── link_functions/   # C++ link function implementations
│   └── optimizers/       # Loss functions and optimizers
├── dataloader/            # Data loading utilities
│   ├── dataloader.h      # C++ dataloader interface
│   └── dataloader.cpp    # CSV loading implementation
├── tests/                 # Test files and examples
│   ├── tensor_test.py    # Tensor operations tests
│   ├── glm_test.py       # GLM usage examples
│   └── svm_test.py       # SVM usage examples
└── build/                 # Build artifacts (generated)
```

## Installation

### Prerequisites

- Python 3.7+
- C++11 compatible compiler (g++, clang++)
- CMake 3.12+
- pybind11

### Building from Source

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tensorkit-learn.git
cd tensorkit-learn
```

2. Create a build directory:
```bash
mkdir build
cd build
```

3. Build the C++ components and Python bindings:
```bash
cmake ..
make -j$(nproc)
```

4. Install the Python module:
```bash
cd ..
pip install -e .
```

## Usage

### Basic Tensor Operations

```python
import tensor_slow

# Create tensors
a = tensor_slow.Tensor.zeros([3, 3])
b = tensor_slow.Tensor.ones([3, 3])
c = tensor_slow.Tensor.from_values([[1, 2], [3, 4]], [2, 2])

# Operations
result = a + b  # Element-wise addition
matmul_result = c @ c.transpose()  # Matrix multiplication
```

### Generalized Linear Models (GLM)

```python
from ml_models.glm import GLM
from ml_models.link_functions.link_functions import LogitLink
from ml_models.optimizers.loss_functions import BCELoss
import tensor_slow

# Create synthetic data
X = tensor_slow.Tensor.from_values([[...]], shape)
y = tensor_slow.Tensor.from_values([[...]], shape)

# Initialize GLM for logistic regression
model = GLM(
    link_function=LogitLink(),
    loss_function=BCELoss(),
    learning_rate=0.01
)

# Train
model.fit(X, y, epochs=100)

# Predict
predictions = model.predict(X_test)
```

### Support Vector Machine (SVM)

```python
from ml_models.svm import SVM
from ml_models.kernel_func import rbf_kernel

# Create SVM with RBF kernel
svm = SVM(
    kernel=rbf_kernel,
    kernel_params={'gamma': 0.1},
    C=1.0,
    learning_rate=0.01
)

# Train
svm.fit(X, y, epochs=100)

# Predict
predictions = svm.predict(X_test)
```

### Multi-Layer Perceptron (MLP)

```python
from ml_models.mlp import MLP, create_classifier, create_regressor
from ml_models.optimizers.loss_functions import Binary_CELoss, MSELoss
import tensor_slow as ts

# Method 1: Quick classifier creation
classifier = create_classifier(
    input_dim=4,
    hidden_dims=[8, 4],
    num_classes=1,
    loss_function=Binary_CELoss()
)

# Method 2: Manual network construction
mlp = MLP()
mlp.add_layer(4, 8, activation='relu')
mlp.add_layer(8, 4, activation='relu') 
mlp.add_layer(4, 1, activation='sigmoid')
mlp.loss_function = Binary_CELoss()

# Train the model
X = ts.Tensor.from_values([100, 4], data)  # 100 samples, 4 features
y = ts.Tensor.from_values([100, 1], labels) # 100 labels
mlp.fit(X, y, epochs=100, learning_rate=0.01)

# Make predictions
predictions = mlp.predict(X_test)
class_predictions = mlp.predict_classes(X_test, threshold=0.5)
```

### Regression with MLP

```python
# Create regression model
regressor = create_regressor(
    input_dim=3,
    hidden_dims=[16, 8],
    output_dim=1,
    loss_function=MSELoss()
)

# Train
regressor.fit(X_train, y_train, epochs=200, learning_rate=0.001)

# Predict
predictions = regressor.predict(X_test)
```

### Custom Loss Functions

```python
from ml_models.optimizers.loss_functions import Loss
import tensor_slow

class CustomLoss(Loss):
    def forward(self, y_true, y_pred):
        # Implement forward pass
        return loss_tensor
    
    def backward(self, y_true, y_pred):
        # Implement gradient computation
        return gradient_tensor
```

## Components

### Tensor Operations
- Basic arithmetic: `+`, `-`, `*`, `/`
- Matrix operations: `@` (matmul), `transpose()`, `inverse()`
- Element-wise functions: `log()`, `exp()`, `clamp()`
- Creation methods: `zeros()`, `ones()`, `from_values()`

### Machine Learning Models
- **GLM**: Supports various link functions (identity, logit, log) for different regression types
- **SVM**: Kernel-based classification with linear, polynomial, and RBF kernels
- **MLP**: Multi-layer perceptron with customizable architecture and activation functions

### Optimizers and Loss Functions
- Binary Cross-Entropy (BCE) Loss
- Mean Squared Error (MSE) Loss
- Gradient-based optimization with configurable learning rates

### Link Functions (C++)
- IdentityLink: For linear regression
- LogitLink: For logistic regression
- LogLink: For Poisson regression

## Development Status

### Completed
- ✅ Core tensor operations in C++
- ✅ Python bindings for tensor library
- ✅ Generalized Linear Models (GLM)
- ✅ Support Vector Machines with kernel methods
- ✅ Loss functions (BCE, MSE)
- ✅ Link functions for GLM
- ✅ Basic data loader for CSV files
- ✅ Multi-layer Perceptron (MLP) with C++ implementation
- ✅ Activation functions (ReLU, Sigmoid, Tanh, Linear)

### In Progress
- 🚧 Decision Trees / Gradient Boosting

### Future Work
- 🔮 Convolutional operations
- 🔮 Recurrent neural networks
- 🔮 Advanced optimizers (Adam, RMSprop)
- 🔮 GPU acceleration

## Testing

Run the test suite to verify installation:

```bash
# Test tensor operations
python tests/tensor_test.py

# Test GLM
python tests/glm_test.py

# Test SVM
python tests/svm_test.py

# Test MLP
python tests/mlp_test.py
```

## Contributing

This is an educational project aimed at understanding ML fundamentals. Contributions that enhance clarity, add educational value, or implement new algorithms from scratch are welcome.

## License

[Add your license here]

## Acknowledgments

Built as a learning project to understand the internals of modern ML frameworks like PyTorch and TensorFlow.