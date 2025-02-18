import tensor_slow as ts
from svm import SVM

# Define a small dataset manually:
X_values = [1, 2, 2, 3, 3, 3, 4, 5]  # 4 samples, 2 features
y_values = [1, 1, -1, -1]             # 4 labels

X_tensor = ts.Tensor.from_values([4, 2], X_values)
y_tensor = ts.Tensor.from_values([4, 1], y_values)

svm = SVM(learning_rate=0.01, lambda_reg=0.01, epochs=1000)
svm.fit(X_tensor, y_tensor)

# Test on a small test set:
test_X_values = [1, 2, 3, 4]  # 2 samples, 2 features
test_X_tensor = ts.Tensor.from_values([2, 2], test_X_values)
predictions = svm.predict(test_X_tensor)

print("Predictions:")
predictions.print()
