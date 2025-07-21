#!/usr/bin/env python3
"""
Test file for MLP implementation with classification and regression examples.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tensor_slow as ts
from mlp import MLP, create_classifier, create_regressor
from optimizers.loss_functions import Binary_CELoss, MSELoss
import math


def create_synthetic_classification_data():
    """Create synthetic binary classification data."""
    print("Creating synthetic classification data...")
    
    # Simple 2D classification problem
    # Class 0: points around (0, 0)
    # Class 1: points around (2, 2)
    
    # Class 0 data
    class0_data = [
        [0.1, 0.2], [0.3, -0.1], [-0.2, 0.3], [0.0, 0.0],
        [0.2, 0.1], [-0.1, -0.2], [0.4, 0.0], [-0.3, 0.2]
    ]
    
    # Class 1 data
    class1_data = [
        [1.9, 2.1], [2.2, 1.8], [1.8, 2.0], [2.0, 2.0],
        [1.7, 2.2], [2.3, 1.9], [1.9, 1.8], [2.1, 2.2]
    ]
    
    # Combine data
    X_data = class0_data + class1_data
    y_data = [[0.0]] * len(class0_data) + [[1.0]] * len(class1_data)
    
    # Create tensors
    X = ts.Tensor.from_values([len(X_data), 2], [val for row in X_data for val in row])
    y = ts.Tensor.from_values([len(y_data), 1], [val for row in y_data for val in row])
    
    print(f"Created {len(X_data)} samples with 2 features")
    return X, y


def create_synthetic_regression_data():
    """Create synthetic regression data."""
    print("Creating synthetic regression data...")
    
    # Simple 1D regression: y = 2*x + 1 + noise
    X_data = []
    y_data = []
    
    for i in range(20):
        x = i * 0.1 - 1.0  # x in [-1, 1]
        y = 2.0 * x + 1.0 + 0.1 * (i % 3 - 1)  # Add small noise
        X_data.append([x])
        y_data.append([y])
    
    # Create tensors
    X = ts.Tensor.from_values([len(X_data), 1], [val for row in X_data for val in row])
    y = ts.Tensor.from_values([len(y_data), 1], [val for row in y_data for val in row])
    
    print(f"Created {len(X_data)} samples with 1 feature")
    return X, y


def test_manual_mlp_construction():
    """Test manual MLP construction with individual layer additions."""
    print("\n=== Testing Manual MLP Construction ===")
    
    # Create MLP manually
    mlp = MLP()
    mlp.add_layer(2, 4, activation='relu')    # Input to hidden
    mlp.add_layer(4, 2, activation='relu')    # Hidden to hidden
    mlp.add_layer(2, 1, activation='sigmoid') # Hidden to output
    
    print(f"Created MLP with {mlp.get_num_layers()} layers")
    
    # Test forward pass with dummy data
    X = ts.Tensor.from_values([1, 2], [0.5, -0.3])
    output = mlp.forward(X)
    
    print("Forward pass successful")
    print(f"Input shape: {X.getShape()}")
    print(f"Output shape: {output.getShape()}")
    print(f"Output value: {output[[0, 0]]}")


def test_classification():
    """Test MLP on binary classification task."""
    print("\n=== Testing Binary Classification ===")
    
    # Create data
    X, y = create_synthetic_classification_data()
    
    # Create classifier
    classifier = create_classifier(
        input_dim=2,
        hidden_dims=[8, 4],
        num_classes=1,
        loss_function=Binary_CELoss()
    )
    
    print(f"Created classifier with {classifier.get_num_layers()} layers")
    
    # Test initial predictions
    print("Testing initial predictions...")
    initial_preds = classifier.predict(X)
    print(f"Initial prediction shape: {initial_preds.getShape()}")
    print(f"Sample initial predictions: {initial_preds[[0, 0]]:.4f}, {initial_preds[[1, 0]]:.4f}")
    
    # Train the model
    print("Training classifier...")
    classifier.fit(
        X, y,
        epochs=50,
        learning_rate=0.1,
        verbose=True
    )
    
    # Test final predictions
    print("Testing final predictions...")
    final_preds = classifier.predict(X)
    print(f"Sample final predictions: {final_preds[[0, 0]]:.4f}, {final_preds[[1, 0]]:.4f}")
    
    # Test class predictions
    class_preds = classifier.predict_classes(X, threshold=0.5)
    print(f"Sample class predictions: {class_preds[[0, 0]]}, {class_preds[[1, 0]]}")


def test_regression():
    """Test MLP on regression task."""
    print("\n=== Testing Regression ===")
    
    # Create data
    X, y = create_synthetic_regression_data()
    
    # Create regressor
    regressor = create_regressor(
        input_dim=1,
        hidden_dims=[8, 4],
        output_dim=1,
        loss_function=MSELoss()
    )
    
    print(f"Created regressor with {regressor.get_num_layers()} layers")
    
    # Test initial predictions
    print("Testing initial predictions...")
    initial_preds = regressor.predict(X)
    print(f"Initial prediction shape: {initial_preds.getShape()}")
    print(f"Sample initial predictions: {initial_preds[[0, 0]]:.4f}, {initial_preds[[1, 0]]:.4f}")
    print(f"Sample targets: {y[[0, 0]]:.4f}, {y[[1, 0]]:.4f}")
    
    # Train the model
    print("Training regressor...")
    regressor.fit(
        X, y,
        epochs=100,
        learning_rate=0.01,
        verbose=True
    )
    
    # Test final predictions
    print("Testing final predictions...")
    final_preds = regressor.predict(X)
    print(f"Sample final predictions: {final_preds[[0, 0]]:.4f}, {final_preds[[1, 0]]:.4f}")
    print(f"Sample targets: {y[[0, 0]]:.4f}, {y[[1, 0]]:.4f}")


def test_activations():
    """Test different activation functions."""
    print("\n=== Testing Activation Functions ===")
    
    # Test each activation with simple data
    X = ts.Tensor.from_values([1, 2], [0.5, -0.3])
    
    activations = ['relu', 'sigmoid', 'tanh', 'linear']
    
    for activation in activations:
        print(f"\nTesting {activation} activation:")
        
        mlp = MLP()
        mlp.add_layer(2, 3, activation=activation)
        mlp.add_layer(3, 1, activation='linear')
        
        output = mlp.forward(X)
        print(f"  Input: [{X[[0, 0]]:.3f}, {X[[0, 1]]:.3f}]")
        print(f"  Output: {output[[0, 0]]:.3f}")


def main():
    """Run all tests."""
    print("Starting MLP Tests")
    print("=" * 50)
    
    try:
        # Test basic construction
        test_manual_mlp_construction()
        
        # Test activation functions
        test_activations()
        
        # Test classification
        test_classification()
        
        # Test regression
        test_regression()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)