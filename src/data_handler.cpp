#include "data_handler.hpp"

data_handler::data_handler() {
    data_array = new std::vector<data *>;
    train_array = new std::vector<data *>;
    test_array = new std::vector<data *>;
    validation_array = new std::vector<data *>;
}

data_handler::~data_handler() {
    // TODO: free memory if needed
}

void data_handler::read_feature_vector(std::string path) {
    uint32_t header[4]; // MAGIC | NUM_IMAGES | ROW_SIZE | COL_SIZE
    unsigned char bytes[4];
    FILE *f = fopen(path.c_str(), "rb");
    if (f) {
        for (int i = 0; i < 4; i++) {
            if (fread(bytes, sizeof(bytes), 1, f)) {
                header[i] = convert_to_little_endian(bytes);
            }
        }
        printf("Done getting image file header.\n");
        int image_size = header[2] * header[3];
        for (int i = 0; i < (int)header[1]; i++) {
            data *d = new data();
            uint8_t element[1];
            for (int j = 0; j < image_size; j++) {
                if (fread(element, sizeof(element), 1, f)) {
                    d->append_to_feature_vector(element[0]);
                } else {
                    printf("Error reading from image file.\n");
                    exit(1);
                }
            }
            data_array->push_back(d);
        }
        printf("Successfully read and stored %lu feature vectors.\n", data_array->size());
        fclose(f);
    } else {
        printf("Could not open image file: %s\n", path.c_str());
        exit(1);
    }
}

void data_handler::read_feature_label(std::string path) {
    uint32_t header[2]; // MAGIC | NUM_ITEMS
    unsigned char bytes[4];
    FILE *f = fopen(path.c_str(), "rb");
    if (f) {
        for (int i = 0; i < 2; i++) {
            if (fread(bytes, sizeof(bytes), 1, f)) {
                header[i] = convert_to_little_endian(bytes);
            }
        }
        printf("Done getting label file header.\n");
        for (int i = 0; i < (int)header[1]; i++) {
            uint8_t element[1];
            if (fread(element, sizeof(element), 1, f)) {
                data_array->at(i)->set_label(element[0]);
            } else {
                printf("Error reading from label file.\n");
                exit(1);
            }
        }
        printf("Successfully read and stored %lu labels.\n", data_array->size());
        fclose(f);
    } else {
        printf("Could not open label file: %s\n", path.c_str());
        exit(1);
    }
}

void data_handler::split_data() {
    std::unordered_set<int> used_indexes;
    int train_size = (int)(data_array->size() * TRAIN_SET_PERCENT);
    int test_size = (int)(data_array->size() * TEST_SET_PERCENT);
    int valid_size = (int)(data_array->size() * VALIDATION_SET_PERCENT);

    // Training Data
    int train_count = 0;
    while (train_count < train_size) {
        int rand_index = rand() % data_array->size();
        if (used_indexes.find(rand_index) == used_indexes.end()) {
            train_array->push_back(data_array->at(rand_index));
            used_indexes.insert(rand_index);
            train_count++;
        }
    }

    // Test Data
    int test_count = 0;
    while (test_count < test_size) {
        int rand_index = rand() % data_array->size();
        if (used_indexes.find(rand_index) == used_indexes.end()) {
            test_array->push_back(data_array->at(rand_index));
            used_indexes.insert(rand_index);
            test_count++;
        }
    }

    // Validation Data
    int valid_count = 0;
    while (valid_count < valid_size) {
        int rand_index = rand() % data_array->size();
        if (used_indexes.find(rand_index) == used_indexes.end()) {
            validation_array->push_back(data_array->at(rand_index));
            used_indexes.insert(rand_index);
            valid_count++;
        }
    }

    printf("Training Data Size: %lu\n", train_array->size());
    printf("Test Data Size: %lu\n", test_array->size());
    printf("Validation Data Size: %lu\n", validation_array->size());
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
    printf("Successfully extracted %d unique classes.\n", num_classes);
}

uint32_t data_handler::convert_to_little_endian(const unsigned char *bytes) {
    return (uint32_t)((bytes[0] << 24) |
                      (bytes[1] << 16) |
                      (bytes[2] << 8) |
                      (bytes[3]));
}

std::vector<data *> *data_handler::get_training_data() {
    return train_array;
}

std::vector<data *> *data_handler::get_test_data() {
    return test_array;
}

std::vector<data *> *data_handler::get_validation_data() {
    return validation_array;
}

int main() {
    data_handler *dh = new data_handler();
    dh->read_feature_vector("../FILE_NAME");
    dh->read_feature_label("../FILE_NAME");
    dh->split_data();
    dh->count_classes();
}