#ifndef DATA_HANDLER_HPP
#define DATA_HANDLER_HPP

#include <vector>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include "data.hpp"

class data_handler {
    std::vector<data*>* data_array;
    std::vector<data*>* training_data;
    std::vector<data*>* test_data;
    std::vector<data*>* validation_data;

    int num_classes;
    int feature_vector_size;
    std::map<uint8_t, int> class_map;

    const double TRAIN_SET_PERCENT = 0.75;
    const double TEST_SET_PERCENT = 0.20;
    const double VALIDATION_SET_PERCENT = 0.05;

public:
    data_handler();
    ~data_handler();

    void read_feature_vector(std::string path);
    void read_feature_label(std::string path);
    void split_data();
    void count_classes();

    uint32_t convert_to_little_endian(const unsigned char* bytes);

    std::vector<data*>* get_training_data();
    std::vector<data*>* get_test_data();
    std::vector<data*>* get_validation_data();
};

#endif // DATA_HANDLER_HPP