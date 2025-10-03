#include "data_handler.hpp"
#include <fstream>
#include <random>
#include <stdexcept>
#include <iostream>

data_handler::data_handler() {
    data_array = new std::vector<data*>;
    training_data = new std::vector<data*>;
    test_data = new std::vector<data*>;
    validation_data = new std::vector<data*>;
}

data_handler::~data_handler() {
    // Giải phóng từng con trỏ data trong các vector
    if (data_array) {
        for (data* d : *data_array) {
            delete d;
        }
        delete data_array;
    }
    if (training_data) {
        for (data* d : *training_data) {
            delete d;
        }
        delete training_data;
    }
    if (test_data) {
        for (data* d : *test_data) {
            delete d;
        }
        delete test_data;
    }
    if (validation_data) {
        for (data* d : *validation_data) {
            delete d;
        }
        delete validation_data;
    }
}

void data_handler::read_feature_vector(std::string path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open image file: " + path);
    }

    uint32_t header[4]; // MAGIC | NUM_IMAGES | ROW_SIZE | COL_SIZE
    file.read(reinterpret_cast<char*>(header), sizeof(header));
    for (auto& h : header) {
        h = convert_to_little_endian(reinterpret_cast<unsigned char*>(&h));
    }
    std::cout << "Done getting image file header.\n";

    int image_size = header[2] * header[3];
    for (int i = 0; i < static_cast<int>(header[1]); i++) {
        data* d = new data();
        for (int j = 0; j < image_size; j++) {
            uint8_t element;
            if (!file.read(reinterpret_cast<char*>(&element), sizeof(element))) {
                throw std::runtime_error("Error reading from image file.");
            }
            d->append_to_feature_vector(element);
        }
        data_array->push_back(d);
    }
    std::cout << "Successfully read and stored " << data_array->size() << " feature vectors.\n";
}

void data_handler::read_feature_label(std::string path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open label file: " + path);
    }

    uint32_t header[2]; // MAGIC | NUM_ITEMS
    file.read(reinterpret_cast<char*>(header), sizeof(header));
    for (auto& h : header) {
        h = convert_to_little_endian(reinterpret_cast<unsigned char*>(&h));
    }
    std::cout << "Done getting label file header.\n";

    for (int i = 0; i < static_cast<int>(header[1]); i++) {
        uint8_t element;
        if (!file.read(reinterpret_cast<char*>(&element), sizeof(element))) {
            throw std::runtime_error("Error reading from label file.");
        }
        data_array->at(i)->set_label(element);
    }
    std::cout << "Successfully read and stored " << data_array->size() << " labels.\n";
}

void data_handler::split_data() {
    std::unordered_set<int> used_indexes;
    int train_size = static_cast<int>(data_array->size() * TRAIN_SET_PERCENT);
    int test_size = static_cast<int>(data_array->size() * TEST_SET_PERCENT);
    int valid_size = static_cast<int>(data_array->size() * VALIDATION_SET_PERCENT);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, data_array->size() - 1);

    int train_count = 0;
    while (train_count < train_size) {
        int rand_index = dist(gen);
        if (used_indexes.find(rand_index) == used_indexes.end()) {
            training_data->push_back(data_array->at(rand_index));
            used_indexes.insert(rand_index);
            train_count++;
        }
    }

    int test_count = 0;
    while (test_count < test_size) {
        int rand_index = dist(gen);
        if (used_indexes.find(rand_index) == used_indexes.end()) {
            test_data->push_back(data_array->at(rand_index));
            used_indexes.insert(rand_index);
            test_count++;
        }
    }

    int valid_count = 0;
    while (valid_count < valid_size) {
        int rand_index = dist(gen);
        if (used_indexes.find(rand_index) == used_indexes.end()) {
            validation_data->push_back(data_array->at(rand_index));
            used_indexes.insert(rand_index);
            valid_count++;
        }
    }

    std::cout << "Training Data Size: " << training_data->size() << "\n";
    std::cout << "Test Data Size: " << test_data->size() << "\n";
    std::cout << "Validation Data Size: " << validation_data->size() << "\n";
}

void data_handler::count_classes() {
    int count = 0;
    for (unsigned i = 0; i < data_array->size(); i++) {
        if (class_map.find(data_array->at(i)->get_label()) == class_map.end()) {
            class_map[data_array->at(i)->get_label()] = count;
            data_array->at(i)->set_enumerated_label(count);
            count++;
        }
    }
    num_classes = count;
    std::cout << "Successfully extracted " << num_classes << " unique classes.\n";
}

uint32_t data_handler::convert_to_little_endian(const unsigned char* bytes) {
    return (uint32_t)((bytes[0] << 24) |
                      (bytes[1] << 16) |
                      (bytes[2] << 8) |
                      (bytes[3]));
}

std::vector<data*>* data_handler::get_training_data() {
    return training_data;
}

std::vector<data*>* data_handler::get_test_data() {
    return test_data;
}

std::vector<data*>* data_handler::get_validation_data() {
    return validation_data;
}