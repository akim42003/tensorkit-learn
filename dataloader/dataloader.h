//dataloader header
#ifndef DATALOADER_H
#define DATALOADER_H

#include <vector>
#include <string>
#include "../tensor_cpp/tensor.h" // Include your Tensor class header

class DataLoader {
private: 
    std::vector<Tensor> data;      // Store data as Tensors
    size_t batch_size;             // Batch size for loading
    bool shuffle;                  // Whether to shuffle the data
    std::vector<size_t> indices;   // Indices for shuffling

    void createIndices();          // Initialize indices for shuffling
    void shuffleIndices();         // Shuffle the indices

public:
    // Constructor
    DataLoader(const std::string& file_path, const std::vector<size_t>& tensor_shape, size_t batch_size = 32, bool shuffle = true);

    // Load data from a CSV file
    void loadData(const std::string& file_path, const std::vector<size_t>& tensor_shape);

    // Fetch the next batch of data
    std::vector<Tensor> getNextBatch(size_t start_index);

    // Iterate through the dataset
    void iterateDataset();
};

#endif // DATALOADER_H
