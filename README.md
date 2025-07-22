# tensorkit-learn

A tensor-based machine learning library built from scratch for educational purposes. This project implements fundamental ML algorithms using a custom C++ tensor library with Python bindings, providing a deep understanding of how modern ML frameworks work under the hood.

## Overview

tensorkit-learn is a hybrid C++/Python machine learning framework that demonstrates the implementation of core ML concepts from first principles. It features a custom tensor library written in C++ for performance, with Python bindings for ease of use and high-level algorithm implementation.

### Key Features

- **Custom Tensor Library**: N-dimensional array operations implemented in C++ with full broadcasting support
- **Machine Learning Models**:
  - Generalized Linear Models (GLM) - Classification & Regression
  - Multi-Layer Perceptrons (MLP) - Classification & Regression
  - Support Vector Machines (SVM) - Multiple kernel support
- **Data Loading**: Robust CSV data loader with batching and shuffling
- **Modular Architecture**: Separate components for tensors, optimizers, loss functions, and models
- **Python Bindings**: Seamless integration between C++ performance and Python flexibility

## Architecture

The project follows a layered architecture:

```
┌─────────────────────────────────────────┐
│         Python ML Models                │
│    (GLM, SVM, MLP, Optimizers)          │
├─────────────────────────────────────────┤
│      Python Bindings (pybind11)         │
│   (Tensor, DataLoader, Link Functions,  │
│           MLP Bindings)                 │
├─────────────────────────────────────────┤
│         C++ Core Libraries              │
│ (Tensor, Link Functions, DataLoader,    │
│            MLP Network)                 │
└─────────────────────────────────────────┘
```

## Project Structure

```
tensorkit-learn/
├── tensor_cpp/             # Core C++ tensor implementation
│   ├── tensor.h           # Tensor class definition
│   ├── tensor.cpp         # Tensor operations implementation
│   └── CMakeLists.txt     # Build configuration for tensor library
├── bindings/              # Python-C++ bindings
│   ├── bindings.cpp       # Main tensor bindings
│   ├── dl_bindings.cpp    # DataLoader bindings
│   ├── lf_bindings.cpp    # Link function bindings
│   ├── mlp_bindings.cpp   # MLP bindings
│   └── CMakeLists.txt     # Build configuration for bindings
├── ml_models/             # Machine learning algorithms
│   ├── glm.py            # Generalized Linear Models
│   ├── svm.py            # Support Vector Machine
│   ├── mlp.py            # Multi-Layer Perceptron
│   ├── kernel_func.py    # Kernel functions for SVM
│   ├── glm_test.py       # GLM usage examples
│   ├── svm_test.py       # SVM usage examples
│   ├── mlp_test.py       # MLP usage examples
│   ├── link_functions/   # C++ link function implementations
│   │   ├── link_functions.h    # Link function headers
│   │   ├── link_functions.cpp  # Link function implementations
│   │   └── CMakeLists.txt      # Build configuration
│   ├── mlp/              # C++ MLP implementation
│   │   ├── mlp.h         # MLP class definition
│   │   └── mlp.cpp       # MLP implementation
│   └── optimizers/       # Loss functions and optimizers
│       └── loss_functions.py  # Python loss function implementations
├── dataloader/            # Data loading utilities
│   ├── dataloader.h      # C++ dataloader interface
│   ├── dataloader.cpp    # CSV loading implementation
│   └── CMakeLists.txt    # Build configuration for dataloader
├── tests/                 # Test files and examples
│   ├── tensor_test.py    # Python tensor operations tests
│   ├── tensor_test.cpp   # C++ tensor operations tests
│   ├── lf_test.py        # Link function tests
│   ├── ce_test.py        # Cross-entropy loss tests
│   ├── mse_test.py       # MSE loss tests
│   └── dl_test.cpp       # DataLoader tests
└── build/                 # Build artifacts (generated)
```

## Quick Start

### Prerequisites

- Python 3.9+
- C++11 compatible compiler (g++, clang++)
- CMake 3.12+
- make

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tensorkit-learn.git
cd tensorkit-learn
```

2. Create and activate a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install pybind11
```

4. Build the C++ components and Python bindings:
```bash
./build.sh
```

### Clean Build

To rebuild from scratch:
```bash
./build.sh clean
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
result = a.tplus(b)  # Element-wise addition
matmul_result = b.matmul(a)  # Matrix multiplication
```
### Data Loading from CSV

```python
import dataloader

# Load CSV data (no headers, numeric data only)
dl = dataloader.DataLoader("data.csv", [4], batch_size=32, shuffle=True)

# Iterate through dataset
dl.iterate_dataset()

# Get specific batch
batch = dl.get_next_batch(0)
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

## Components

### Tensor Operations
- Basic arithmetic: `tplus`, `tminus`, `elementwise_multiply`, `divide`, `log`, `exp`
- Matrix operations: `matmul()`, `transpose()`, `inverse()`
- Element-wise functions: `log()`, `exp()`, `clamp()`
- Creation methods: `zeros()`, `ones()`, `from_values()`

### Machine Learning Models
- **GLM**: Supports various link functions (identity, logit, log) for different regression types
- **SVM**: Kernel-based classification with linear, polynomial, and RBF kernels
- **MLP**: Multi-layer perceptron with customizable architecture and activation functions

### Optimizers and Loss Functions
- Binary Cross-Entropy (BCE) Loss
- Mean Squared Error (MSE) Loss

### Link Functions (C++)
- IdentityLink: For linear regression
- LogitLink: For logistic regression
- LogLink: For Poisson regression

## Development Status

### Completed ✅
- Core tensor operations in C++
- Python bindings for all components
- Generalized Linear Models (GLM)
- Multi-Layer Perceptrons (MLP) with C++ backend
- Support Vector Machines with kernel methods
- CSV data loader with batching/shuffling
- Comprehensive test suite with performance evaluation
- Loss functions (BCE, MSE) and link functions
- Activation functions (ReLU, Sigmoid, Tanh, Linear)

# Future Work
- Convolutional operations
- Gradient Boosting and Tree models
- Advanced optimizers (Adam, RMSprop)
- GPU acceleration

## Testing

Run the test suite to verify installation:

### Quick Test
```bash
# Run complete test suite with CSV data
python comprehensive_ml_test.py
```

### Individual Component Tests
```bash
# Test individual models
python ml_models/glm_test.py
python ml_models/mlp_test.py
python ml_models/svm_test.py

# Test core components
python tests/tensor_test.py
python tests/lf_test.py
python tests/mse_test.py
python tests/ce_test.py
```

### Test Coverage

The comprehensive test suite includes:
- ✅ **DataLoader**: CSV loading, batching, shuffling functionality
- ✅ **GLM**: Classification (LogitLink) and regression (IdentityLink)
- ✅ **MLP**: Classification and regression with multi-layer networks
- ✅ **SVM**: Linear, polynomial, and RBF kernel classification
- ✅ **Performance**: Accuracy and MSE evaluation across models

### Test Data

Sample CSV datasets included:
- `test_data_classification.csv` - 16 samples, 3 features, binary classification
- `test_data_regression.csv` - 16 samples, 2 features, continuous regression
- `test_data_no_header.csv` - Headerless CSV for direct dataloader testing

## CSV Data Format

The dataloader expects CSV files with:
- Numeric data only (no headers)
- Comma-separated values
- Consistent number of columns per row

Example CSV format:
```csv
0.1,0.2,0.3,0
0.3,-0.1,0.5,0
1.9,2.1,1.8,1
```

## Acknowledgments

Built as a learning project to understand the internals of modern ML frameworks like PyTorch and TensorFlow.
