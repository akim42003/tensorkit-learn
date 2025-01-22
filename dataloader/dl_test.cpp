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

        // Step 4: Test specific batches
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

        // Step 5: Confirm shuffle (if enabled)
        if (loader.getNextBatch(0) != batch1) {
            std::cout << "Shuffling verified: Data order changed.\n";
        } else {
            std::cout << "Shuffle disabled or data not shuffled.\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << "\n";
    }

    // Step 6: Cleanup
    std::remove(test_file.c_str());
}

int main() {
    testDataLoader();
    return 0;
}
