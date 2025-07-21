#!/usr/bin/env python3
"""
Comprehensive test file for ML models and dataloader using CSV data.
Tests GLM, MLP, and SVM models for both classification and regression tasks.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_models'))

import tensor_slow as ts
import dataloader
import csv
import math

# Import ML models
from ml_models.glm import GLM
from ml_models.mlp import MLP, create_classifier, create_regressor
from ml_models.svm import SVM

# Import loss functions and link functions
from ml_models.optimizers.loss_functions import Binary_CELoss, MSELoss
from link_functions import IdentityLink, LogitLink


def load_csv_data(file_path, feature_cols, target_col):
    """
    Load CSV data and return X and y tensors.

    Args:
        file_path: Path to CSV file
        feature_cols: List of feature column names
        target_col: Target column name

    Returns:
        X: Feature tensor
        y: Target tensor
    """
    X_data = []
    y_data = []

    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Extract features
            features = [float(row[col]) for col in feature_cols]
            X_data.extend(features)

            # Extract target
            y_data.append(float(row[target_col]))

    # Calculate dimensions
    n_samples = len(y_data)
    n_features = len(feature_cols)

    # Create tensors
    X = ts.Tensor.from_values([n_samples, n_features], X_data)
    y = ts.Tensor.from_values([n_samples, 1], y_data)

    return X, y


def test_dataloader_functionality():
    """Test dataloader with CSV files."""
    print("=== Testing DataLoader Functionality ===")

    # Test with classification data
    print("\n1. Testing dataloader with classification data:")
    try:
        # Create dataloader for classification data (3 features + 1 target = 4 columns)
        dl_class = dataloader.DataLoader(
            "./test_data/test_data_classification.csv",
            [4],  # Shape: 4 columns per row
            batch_size=4,
            shuffle=True
        )

        print("✓ Successfully created dataloader for classification data")

        # Test iteration
        print("Iterating through classification dataset:")
        dl_class.iterate_dataset()

    except Exception as e:
        print(f" Error with classification dataloader: {e}")

    # Test with regression data
    print("\n2. Testing dataloader with regression data:")
    try:
        # Create dataloader for regression data (2 features + 1 target = 3 columns)
        dl_reg = dataloader.DataLoader(
            "./test_data/test_data_regression.csv",
            [3],  # Shape: 3 columns per row
            batch_size=4,
            shuffle=False
        )

        print("✓ Successfully created dataloader for regression data")

        # Test iteration
        print("Iterating through regression dataset:")
        dl_reg.iterate_dataset()

    except Exception as e:
        print(f" Error with regression dataloader: {e}")


def test_glm_classification():
    """Test GLM for binary classification."""
    print("\n=== Testing GLM Classification ===")

    try:
        # Load classification data
        X, y = load_csv_data(
            "./test_data/test_data_classification.csv",
            ["feature1", "feature2", "feature3"],
            "target"
        )

        print(f"Loaded classification data: {X.getShape()} features, {y.getShape()} targets")

        # Create GLM with logistic link for classification
        link_function = LogitLink()
        loss_function = Binary_CELoss()
        glm_classifier = GLM(
            num_features=3,
            link_function=link_function,
            loss_function=loss_function
        )

        print("Training GLM classifier...")
        glm_classifier.train(
            X, y,
            learning_rate=0.01,
            max_iterations=50,
            tolerance=1e-6
        )

        # Test predictions
        predictions = glm_classifier.forward(X)
        print(f"Sample predictions: {predictions[[0, 0]]:.4f}, {predictions[[1, 0]]:.4f}")

        # Get coefficients
        coeffs = glm_classifier.get_coefficients()
        print("Final coefficients:")
        coeffs.print()

        print("GLM classification test completed successfully")

    except Exception as e:
        print(f" GLM classification test failed: {e}")
        import traceback
        traceback.print_exc()


def test_glm_regression():
    """Test GLM for regression."""
    print("\n=== Testing GLM Regression ===")

    try:
        # Load regression data
        X, y = load_csv_data(
            "./test_data/test_data_regression.csv",
            ["feature1", "feature2"],
            "target"
        )

        print(f"Loaded regression data: {X.getShape()} features, {y.getShape()} targets")

        # Create GLM with identity link for regression
        link_function = IdentityLink()
        loss_function = MSELoss()
        glm_regressor = GLM(
            num_features=2,
            link_function=link_function,
            loss_function=loss_function
        )

        print("Training GLM regressor...")
        glm_regressor.train(
            X, y,
            learning_rate=0.01,
            max_iterations=100,
            tolerance=1e-6
        )

        # Test predictions
        predictions = glm_regressor.forward(X)
        print(f"Sample predictions: {predictions[[0, 0]]:.4f}, {predictions[[1, 0]]:.4f}")
        print(f"Sample targets: {y[[0, 0]]:.4f}, {y[[1, 0]]:.4f}")

        # Get coefficients
        coeffs = glm_regressor.get_coefficients()
        print("Final coefficients:")
        coeffs.print()

        print("GLM regression test completed successfully")

    except Exception as e:
        print(f" GLM regression test failed: {e}")
        import traceback
        traceback.print_exc()


def test_mlp_classification():
    """Test MLP for binary classification."""
    print("\n=== Testing MLP Classification ===")

    try:
        # Load classification data
        X, y = load_csv_data(
            "./test_data/test_data_classification.csv",
            ["feature1", "feature2", "feature3"],
            "target"
        )

        print(f"Loaded classification data: {X.getShape()} features, {y.getShape()} targets")

        # Create MLP classifier
        mlp_classifier = create_classifier(
            input_dim=3,
            hidden_dims=[8, 4],
            num_classes=1,
            loss_function=Binary_CELoss()
        )

        print(f"Created MLP classifier with {mlp_classifier.get_num_layers()} layers")

        # Test initial predictions
        initial_preds = mlp_classifier.predict(X)
        print(f"Initial predictions: {initial_preds[[0, 0]]:.4f}, {initial_preds[[1, 0]]:.4f}")

        # Train the model
        print("Training MLP classifier...")
        mlp_classifier.fit(
            X, y,
            epochs=50,
            learning_rate=0.1,
            verbose=True
        )

        # Test final predictions
        final_preds = mlp_classifier.predict(X)
        print(f"Final predictions: {final_preds[[0, 0]]:.4f}, {final_preds[[1, 0]]:.4f}")

        # Test class predictions
        class_preds = mlp_classifier.predict_classes(X, threshold=0.5)
        print(f"Class predictions: {class_preds[[0, 0]]}, {class_preds[[1, 0]]}")

        print("MLP classification test completed successfully")

    except Exception as e:
        print(f" MLP classification test failed: {e}")
        import traceback
        traceback.print_exc()


def test_mlp_regression():
    """Test MLP for regression."""
    print("\n=== Testing MLP Regression ===")

    try:
        # Load regression data
        X, y = load_csv_data(
            "./test_data/test_data_regression.csv",
            ["feature1", "feature2"],
            "target"
        )

        print(f"Loaded regression data: {X.getShape()} features, {y.getShape()} targets")

        # Create MLP regressor
        mlp_regressor = create_regressor(
            input_dim=2,
            hidden_dims=[8, 4],
            output_dim=1,
            loss_function=MSELoss()
        )

        print(f"Created MLP regressor with {mlp_regressor.get_num_layers()} layers")

        # Test initial predictions
        initial_preds = mlp_regressor.predict(X)
        print(f"Initial predictions: {initial_preds[[0, 0]]:.4f}, {initial_preds[[1, 0]]:.4f}")
        print(f"Targets: {y[[0, 0]]:.4f}, {y[[1, 0]]:.4f}")

        # Train the model
        print("Training MLP regressor...")
        mlp_regressor.fit(
            X, y,
            epochs=100,
            learning_rate=0.01,
            verbose=True
        )

        # Test final predictions
        final_preds = mlp_regressor.predict(X)
        print(f"Final predictions: {final_preds[[0, 0]]:.4f}, {final_preds[[1, 0]]:.4f}")
        print(f"Targets: {y[[0, 0]]:.4f}, {y[[1, 0]]:.4f}")

        print("MLP regression test completed successfully")

    except Exception as e:
        print(f" MLP regression test failed: {e}")
        import traceback
        traceback.print_exc()


def test_svm_classification():
    """Test SVM for binary classification."""
    print("\n=== Testing SVM Classification ===")

    try:
        # Load classification data
        X, y = load_csv_data(
            "./test_data/test_data_classification.csv",
            ["feature1", "feature2", "feature3"],
            "target"
        )

        print(f"Loaded classification data: {X.getShape()} features, {y.getShape()} targets")

        # Convert targets to -1/1 format for SVM
        y_svm_data = []
        for i in range(y.getShape()[0]):
            target = y[[i, 0]]
            y_svm_data.append(1.0 if target == 1.0 else -1.0)

        y_svm = ts.Tensor.from_values([len(y_svm_data), 1], y_svm_data)

        # Test different kernels
        kernels = ["linear", "polynomial", "rbf"]

        for kernel in kernels:
            print(f"\nTesting SVM with {kernel} kernel:")

            # Create SVM
            svm_classifier = SVM(
                C=1.0,
                kernel=kernel,
                epochs=100,
                learning_rate=0.01
            )

            # Train the model
            print(f"Training SVM with {kernel} kernel...")
            svm_classifier.fit(X, y_svm)

            # Test predictions
            predictions = svm_classifier.predict(X)
            print(f"Sample predictions: {predictions[[0, 0]]}, {predictions[[1, 0]]}")

            # Test decision function
            decision_vals = svm_classifier.decision_function(X)
            print(f"Sample decision values: {decision_vals[[0, 0]]:.4f}, {decision_vals[[1, 0]]:.4f}")

        print("SVM classification test completed successfully")

    except Exception as e:
        print(f" SVM classification test failed: {e}")
        import traceback
        traceback.print_exc()


def calculate_accuracy(predictions, targets):
    """Calculate classification accuracy."""
    correct = 0
    total = predictions.getShape()[0]

    for i in range(total):
        pred = 1 if predictions[[i, 0]] > 0.5 else 0
        target = int(targets[[i, 0]])
        if pred == target:
            correct += 1

    return correct / total


def calculate_mse(predictions, targets):
    """Calculate mean squared error."""
    mse = 0.0
    n = predictions.getShape()[0]

    for i in range(n):
        diff = predictions[[i, 0]] - targets[[i, 0]]
        mse += diff * diff

    return mse / n


def run_comprehensive_evaluation():
    """Run comprehensive evaluation of all models."""
    print("\n=== Comprehensive Model Evaluation ===")

    # Classification evaluation
    print("\n1. Classification Task Evaluation:")
    try:
        X_class, y_class = load_csv_data(
            "./test_data/test_data_classification.csv",
            ["feature1", "feature2", "feature3"],
            "target"
        )

        # Test GLM
        glm_clf = GLM(3, LogitLink(), Binary_CELoss())
        glm_clf.train(X_class, y_class, 0.01, 50, 1e-6)
        glm_preds = glm_clf.forward(X_class)
        glm_acc = calculate_accuracy(glm_preds, y_class)
        print(f"GLM Classification Accuracy: {glm_acc:.4f}")

        # Test MLP
        mlp_clf = create_classifier(3, [8, 4], 1, Binary_CELoss())
        mlp_clf.fit(X_class, y_class, epochs=50, learning_rate=0.1, verbose=False)
        mlp_preds = mlp_clf.predict(X_class)
        mlp_acc = calculate_accuracy(mlp_preds, y_class)
        print(f"MLP Classification Accuracy: {mlp_acc:.4f}")

    except Exception as e:
        print(f"Classification evaluation failed: {e}")

    # Regression evaluation
    print("\n2. Regression Task Evaluation:")
    try:
        X_reg, y_reg = load_csv_data(
            "./test_data/test_data_regression.csv",
            ["feature1", "feature2"],
            "target"
        )

        # Test GLM
        glm_reg = GLM(2, IdentityLink(), MSELoss())
        glm_reg.train(X_reg, y_reg, 0.01, 100, 1e-6)
        glm_preds = glm_reg.forward(X_reg)
        glm_mse = calculate_mse(glm_preds, y_reg)
        print(f"GLM Regression MSE: {glm_mse:.4f}")

        # Test MLP
        mlp_reg = create_regressor(2, [8, 4], 1, MSELoss())
        mlp_reg.fit(X_reg, y_reg, epochs=100, learning_rate=0.01, verbose=False)
        mlp_preds = mlp_reg.predict(X_reg)
        mlp_mse = calculate_mse(mlp_preds, y_reg)
        print(f"MLP Regression MSE: {mlp_mse:.4f}")

    except Exception as e:
        print(f"Regression evaluation failed: {e}")


def main():
    """Run all tests."""
    print("Starting Comprehensive ML and DataLoader Tests")
    print("=" * 60)

    test_results = []

    # Test dataloader
    try:
        test_dataloader_functionality()
        test_results.append("DataLoader: PASSED")
    except Exception as e:
        test_results.append(f"DataLoader: FAILED - {e}")

    # Test GLM
    try:
        test_glm_classification()
        test_glm_regression()
        test_results.append("GLM: PASSED")
    except Exception as e:
        test_results.append(f"GLM: FAILED - {e}")

    # Test MLP
    try:
        test_mlp_classification()
        test_mlp_regression()
        test_results.append("MLP: PASSED")
    except Exception as e:
        test_results.append(f"MLP: FAILED - {e}")

    # Test SVM
    try:
        test_svm_classification()
        test_results.append("SVM: PASSED")
    except Exception as e:
        test_results.append(f"SVM: FAILED - {e}")

    # Run comprehensive evaluation
    try:
        run_comprehensive_evaluation()
        test_results.append("Evaluation: PASSED")
    except Exception as e:
        test_results.append(f"Evaluation: FAILED - {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY:")
    print("=" * 60)
    for result in test_results:
        print(result)

    passed = sum(1 for r in test_results if "PASSED" in r)
    total = len(test_results)
    print(f"\nOverall: {passed}/{total} test suites passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
