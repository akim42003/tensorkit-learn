# TensorKit Learn

A comprehensive machine learning library built from scratch in C++ with Python bindings for educational purposes. This project implements fundamental ML algorithms using a custom tensor library, providing deep understanding of modern ML frameworks.

## Features

- **Custom Tensor Library**: N-dimensional array operations implemented in C++ with full broadcasting support
- **Machine Learning Models**:
  - Generalized Linear Models (GLM) - Classification & Regression
  - Multi-Layer Perceptrons (MLP) - Classification & Regression  
  - Support Vector Machines (SVM) - Multiple kernel support
- **Data Loading**: Robust CSV data loader with batching and shuffling
- **Link Functions**: Identity, Logit, and Log link functions for GLM
- **Loss Functions**: MSE, Binary Cross-Entropy
- **Python Bindings**: Seamless integration between C++ performance and Python flexibility
- **Comprehensive Testing**: Full test suite with dummy CSV data and performance evaluation

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
├── tensor_cpp/              # Core C++ tensor implementation
│   ├── tensor.h/cpp         # Tensor operations
│   └── CMakeLists.txt       # Build configuration
├── bindings/                # Python-C++ bindings
│   ├── bindings.cpp         # Main tensor bindings
│   ├── dl_bindings.cpp      # DataLoader bindings
│   ├── lf_bindings.cpp      # Link function bindings
│   ├── mlp_bindings.cpp     # MLP bindings
│   └── CMakeLists.txt       # Build configuration
├── ml_models/               # Machine learning algorithms
│   ├── glm.py              # Generalized Linear Models
│   ├── mlp.py              # Multi-Layer Perceptron
│   ├── svm.py              # Support Vector Machine
│   ├── kernel_func.py      # Kernel functions for SVM
│   ├── link_functions/     # C++ link function implementations
│   ├── mlp/                # C++ MLP implementation
│   └── optimizers/         # Loss functions and optimizers
├── dataloader/             # Data loading utilities
│   ├── dataloader.h/cpp    # CSV loading implementation
│   └── CMakeLists.txt      # Build configuration
├── tests/                  # Individual component tests
├── comprehensive_ml_test.py # Complete test suite
├── test_data_*.csv         # Test datasets
└── README.md
```

## Installation

### Prerequisites

- Python 3.7+
- C++11 compatible compiler (g++, clang++)
- CMake 3.14+
- pybind11

### Build Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd tensorkit-learn
```

2. Install Python dependencies:
```bash
pip install pybind11
```

3. Build the project:
```bash
cd bindings
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

4. Verify installation:
```bash
cd ../..
python3 -c "import tensor_slow, dataloader, link_functions, mlp_cpp; print('All modules imported successfully')"
```

## Usage

### Basic Tensor Operations

```python
import tensor_slow as ts

# Create tensors
a = ts.Tensor.zeros([3, 3])
b = ts.Tensor.ones([3, 3])  
c = ts.Tensor.from_values([2, 2], [1.0, 2.0, 3.0, 4.0])

# Operations
result = a.tplus(b)  # Element-wise addition
matmul_result = c.matmul(c.Tp())  # Matrix multiplication
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

### Machine Learning Models

#### GLM Classification

```python
from ml_models.glm import GLM
from ml_models.optimizers.loss_functions import Binary_CELoss
from link_functions import LogitLink

# Create GLM for binary classification
model = GLM(
    num_features=3,
    link_function=LogitLink(),
    loss_function=Binary_CELoss()
)

# Train the model
model.train(X, y, learning_rate=0.01, max_iterations=100, tolerance=1e-6)
predictions = model.forward(X_test)
```

#### GLM Regression

```python
from ml_models.glm import GLM
from ml_models.optimizers.loss_functions import MSELoss
from link_functions import IdentityLink

# Create GLM for regression
model = GLM(
    num_features=2,
    link_function=IdentityLink(),
    loss_function=MSELoss()
)

model.train(X, y, learning_rate=0.01, max_iterations=100, tolerance=1e-6)
```

#### MLP Classification

```python
from ml_models.mlp import create_classifier
from ml_models.optimizers.loss_functions import Binary_CELoss

# Create MLP classifier
classifier = create_classifier(
    input_dim=3,
    hidden_dims=[8, 4],
    num_classes=1,
    loss_function=Binary_CELoss()
)

# Train the model
classifier.fit(X, y, epochs=100, learning_rate=0.01)
predictions = classifier.predict(X_test)
class_predictions = classifier.predict_classes(X_test, threshold=0.5)
```

#### MLP Regression

```python
from ml_models.mlp import create_regressor
from ml_models.optimizers.loss_functions import MSELoss

# Create MLP regressor
regressor = create_regressor(
    input_dim=2,
    hidden_dims=[8, 4],
    output_dim=1,
    loss_function=MSELoss()
)

regressor.fit(X, y, epochs=100, learning_rate=0.01)
predictions = regressor.predict(X_test)
```

#### SVM Classification

```python
from ml_models.svm import SVM

# Create SVM with different kernels
svm_linear = SVM(C=1.0, kernel="linear")
svm_rbf = SVM(C=1.0, kernel="rbf", gamma=0.1)
svm_poly = SVM(C=1.0, kernel="polynomial", degree=3)

# Train the model (targets should be -1/1)
svm_rbf.fit(X, y)
predictions = svm_rbf.predict(X_test)
decision_values = svm_rbf.decision_function(X_test)
```

## Testing

The project includes comprehensive tests for all components:

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

## Components

### Tensor Operations (C++)
- Basic arithmetic: `tplus()`, `tminus()`, `tmul()`, `tdiv()`
- Matrix operations: `matmul()`, `Tp()` (transpose)
- Element-wise functions: `clamp()`, broadcasted operations
- Creation methods: `zeros()`, `ones()`, `from_values()`

### Machine Learning Models
- **GLM**: Supports identity, logit, and log link functions for different regression types
- **SVM**: Kernel-based classification with linear, polynomial, and RBF kernels
- **MLP**: Multi-layer perceptron with customizable architecture and activation functions

### Loss Functions & Optimizers
- Binary Cross-Entropy Loss
- Mean Squared Error Loss
- Gradient-based optimization with configurable learning rates

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

### Future Work 🔮
- Decision Trees / Gradient Boosting
- Convolutional operations
- Recurrent neural networks
- Advanced optimizers (Adam, RMSprop)
- GPU acceleration

## Performance Notes

From test results:
- **GLM**: Fast convergence for linear relationships
- **MLP**: Effective for non-linear patterns (1.0 classification accuracy, 0.09 regression MSE)
- **SVM**: Multiple kernel support for flexible classification
- **DataLoader**: Efficient batch processing with optional shuffling

## Contributing

This is an educational project aimed at understanding ML fundamentals. Contributions that enhance clarity, add educational value, or implement new algorithms from scratch are welcome.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite to ensure everything works
6. Submit a pull request

## Acknowledgments

Built as a learning project to understand the internals of modern ML frameworks like PyTorch and TensorFlow, with emphasis on educational clarity over optimization.