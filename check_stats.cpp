#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>


int main() {
  std::string y_path =
      R"(D:\icu_dataset\processed\cuda_eval\holdout_y_float32.bin)";
  std::ifstream file(y_path, std::ios::binary);
  if (!file) {
    std::cerr << "Cannot open file\n";
    return 1;
  }

  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);

  size_t count = size / sizeof(float);
  std::vector<float> data(count);
  file.read((char *)data.data(), size);

  double sum = 0;
  float min_val = data[0];
  float max_val = data[0];
  for (float v : data) {
    sum += v;
    if (v < min_val)
      min_val = v;
    if (v > max_val)
      max_val = v;
  }

  double mean = sum / count;
  double var_sum = 0;
  for (float v : data) {
    var_sum += (v - mean) * (v - mean);
  }
  double std = std::sqrt(var_sum / count);

  std::cout << "--- Data Statistics ---\n";
  std::cout << "Count: " << count << "\n";
  std::cout << "Min: " << min_val << "\n";
  std::cout << "Max: " << max_val << "\n";
  std::cout << "Mean: " << mean << "\n";
  std::cout << "Std Dev: " << std << "\n";

  return 0;
}
