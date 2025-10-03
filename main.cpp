#include "data_handler.hpp"
#include <iostream>
#include <fstream>

int main() {
    try {
        // Kiểm tra sự tồn tại của file
        std::string image_file = "C:/Users/ASUS/.vscode/mnist_ml/dataset/train-images-idx3-ubyte";
        std::string label_file = "C:/Users/ASUS/.vscode/mnist_ml/dataset/train-labels-idx1-ubyte";
        
        std::ifstream check_image(image_file, std::ios::binary);
        if (!check_image.is_open()) {
            std::cerr << "Cannot open image file: " << image_file << std::endl;
            return 1;
        }
        check_image.close();

        std::ifstream check_label(label_file, std::ios::binary);
        if (!check_label.is_open()) {
            std::cerr << "Cannot open label file: " << label_file << std::endl;
            return 1;
        }
        check_label.close();

        data_handler dh;
        dh.read_feature_vector(image_file);
        dh.read_feature_label(label_file);
        dh.split_data();
        dh.count_classes();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}