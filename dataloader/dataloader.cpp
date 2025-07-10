// dataloader functions
#include "dataloader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <random>
#include <algorithm>

// Constructor
DataLoader::DataLoader(const std::string& file_path, const std::vector<size_t>& tensor_shape, size_t batch_size, bool shuffle)
    : batch_size(batch_size), shuffle(shuffle) {
    loadData(file_path, tensor_shape);
    createIndices();
}

// Load data from a CSV file
void DataLoader::loadData(const std::string& file_path, const std::vector<size_t>& tensor_shape) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + file_path);
    }

    std::string line;
    bool first_line = true;
    while (std::getline(file, line)) {
        // Skip header row
        if (first_line) {
            first_line = false;
            continue;
        }
        
        std::istringstream line_stream(line);
        std::vector<float> values;
        std::string cell;

        while (std::getline(line_stream, cell, ',')) {
            // Trim whitespace
            cell.erase(0, cell.find_first_not_of(" \t\r\n"));
            cell.erase(cell.find_last_not_of(" \t\r\n") + 1);
            
            // Skip empty cells
            if (cell.empty()) {
                continue;
            }
            
            try {
                values.push_back(std::stof(cell)); // Convert string to float
            } catch (const std::invalid_argument& e) {
                throw std::runtime_error("Invalid float value: '" + cell + "' in file: " + file_path);
            } catch (const std::out_of_range& e) {
                throw std::runtime_error("Float value out of range: '" + cell + "' in file: " + file_path);
            }
        }

        size_t total_size = 1;
        for (size_t dim : tensor_shape) {
            total_size *= dim;
        }

        if (values.size() != total_size) {
            throw std::invalid_argument("Data size in row does not match the specified tensor shape.");
        }

        // Create a Tensor and add it to the dataset
        data.push_back(Tensor::from_values(tensor_shape, values));
    }

    file.close();
}

// Initialize indices for shuffling
void DataLoader::createIndices() {
    indices.resize(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        indices[i] = i;
    }
}

// Shuffle the indices with debug information
void DataLoader::shuffleIndices() {
    std::cout << "Indices before shuffle: ";
    for (size_t i : indices) {
        std::cout << i << " ";
    }
    std::cout << "\n";

    std::random_device rd;
    std::mt19937 generator(rd());
    std::shuffle(indices.begin(), indices.end(), generator);

    std::cout << "Indices after shuffle: ";
    for (size_t i : indices) {
        std::cout << i << " ";
    }
    std::cout << "\n";
}

// Fetch the next batch of data
std::vector<Tensor> DataLoader::getNextBatch(size_t start_index) {
    std::vector<Tensor> batch;
    size_t end_index = std::min(start_index + batch_size, data.size());
    for (size_t i = start_index; i < end_index; ++i) {
        batch.push_back(data[indices[i]]);
    }
    return batch;
}

// Iterate through the dataset with debug information
void DataLoader::iterateDataset() {
    if (shuffle) {
        std::cout << "Shuffling the dataset...\n";
        shuffleIndices();
    } else {
        std::cout << "Skipping shuffle as it is disabled.\n";
    }

    size_t total_batches = (data.size() + batch_size - 1) / batch_size;
    for (size_t i = 0; i < total_batches; ++i) {
        auto batch = getNextBatch(i * batch_size);

        std::cout << "Batch " << i + 1 << " (size: " << batch.size() << "):\n";
        for (const auto& tensor : batch) {
            tensor.print(); // Assuming Tensor::print() prints the tensor data
            std::cout << "\n";
        }
    }
}
