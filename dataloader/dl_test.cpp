#include "dataloader.h"
#include <fstream>
#include <iostream>

// Helper function to create a sample CSV file
void createTestCSV(const std::string& file_path) {
    std::ofstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to create test CSV file.");
    }

    // Write sample data (3x4 tensors, 4 rows)
    file << "1,2,3,4,5,6,7,8,9,10,11,12\n";
    file << "13,14,15,16,17,18,19,20,21,22,23,24\n";
    file << "25,26,27,28,29,30,31,32,33,34,35,36\n";
    file << "37,38,39,40,41,42,43,44,45,46,47,48\n";

    file.close();
}

// Helper function to manually compare two batches of tensors
bool areBatchesEqual(const std::vector<Tensor>& batch1, const std::vector<Tensor>& batch2) {
    if (batch1.size() != batch2.size()) {
        return false;
    }

    for (size_t i = 0; i < batch1.size(); ++i) {
        // Compare shapes
        if (batch1[i].getShape() != batch2[i].getShape()) {
            return false;
        }

        // Compare data
        const auto& shape = batch1[i].getShape();
        size_t total_size = 1;
        for (size_t dim : shape) {
            total_size *= dim;
        }

        // Convert flat index to multi-dimensional index
        std::vector<size_t> strides(shape.size(), 1);
        for (int d = shape.size() - 2; d >= 0; --d) {
            strides[d] = strides[d + 1] * shape[d + 1];
        }

        for (size_t j = 0; j < total_size; ++j) {
            std::vector<size_t> multi_index(shape.size());
            size_t remaining = j;
            for (size_t d = 0; d < shape.size(); ++d) {
                multi_index[d] = remaining / strides[d];
                remaining %= strides[d];
            }

            if (batch1[i].operator()(multi_index) != batch2[i].operator()(multi_index)) {
                return false;
            }
        }
    }

    return true;
}

// Test function for DataLoader
void testDataLoader() {
    // Step 1: Create a test CSV file
    const std::string test_file = "test_data.csv";
    createTestCSV(test_file);

    try {
        // Step 2: Initialize DataLoader
        std::cout << "Initializing DataLoader...\n";
        DataLoader loader(test_file, {3, 4}, 2, true); // 3x4 tensors, batch size of 2, shuffle enabled

        // Debug: Print indices before and after shuffle
        std::cout << "Indices before shuffle:\n";
        loader.iterateDataset(); // Prints initial order and data

        std::cout << "\nVerifying shuffle...\n";
        loader.iterateDataset(); // Prints shuffled order and data

        // Step 3: Test specific batches
        std::cout << "\nFetching specific batches:\n";
        auto batch1 = loader.getNextBatch(0);
        std::cout << "Batch 1:\n";
        for (const auto& tensor : batch1) {
            tensor.print();
            std::cout << "\n";
        }

        auto batch2 = loader.getNextBatch(2);
        std::cout << "Batch 2:\n";
        for (const auto& tensor : batch2) {
            tensor.print();
            std::cout << "\n";
        }

        // Step 4: Verify shuffle manually
        auto shuffled_batch1 = loader.getNextBatch(0);
        if (!areBatchesEqual(batch1, shuffled_batch1)) {
            std::cout << "Shuffling verified: Data order changed.\n";
        } else {
            std::cout << "Shuffle disabled or data not shuffled.\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << "\n";
    }

    // Step 5: Cleanup
    std::remove(test_file.c_str());
}

int main() {
    testDataLoader();
    return 0;
}
