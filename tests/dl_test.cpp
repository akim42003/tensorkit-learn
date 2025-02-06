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

        // Compare elements (multi-dimensional indices logic here)
        const auto& shape = batch1[i].getShape();
        size_t total_size = 1;
        for (size_t dim : shape) {
            total_size *= dim;
        }

        for (size_t j = 0; j < total_size; ++j) {
            std::vector<size_t> multi_index(shape.size());
            size_t remaining = j;
            for (size_t d = 0; d < shape.size(); ++d) {
                multi_index[d] = remaining % shape[d];
                remaining /= shape[d];
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

        // Step 3: Test iterateDataset (prints batches)
        std::cout << "Iterating through dataset:\n";
        loader.iterateDataset();

        // Step 4: Fetch batches before shuffling
        std::cout << "\nFetching batches before shuffle:\n";
        auto batch1_before = loader.getNextBatch(0);
        auto batch2_before = loader.getNextBatch(2);

        // Step 5: Shuffle dataset by reinitializing or iterating again
        std::cout << "\nVerifying shuffle...\n";
        loader.iterateDataset(); // Forces shuffle internally

        // Step 6: Fetch batches after shuffling
        auto batch1_after = loader.getNextBatch(0);
        auto batch2_after = loader.getNextBatch(2);

        // Step 7: Compare batches before and after shuffle
        bool is_shuffled = !(areBatchesEqual(batch1_before, batch1_after) &&
                             areBatchesEqual(batch2_before, batch2_after));

        if (is_shuffled) {
            std::cout << "Shuffling verified: Data order changed.\n";
        } else {
            std::cout << "Shuffle failed: Data order did not change.\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << "\n";
    }

    // Step 8: Cleanup
    std::remove(test_file.c_str());
}


int main() {
    testDataLoader();
    return 0;
}
