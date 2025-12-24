#pragma once
#include <algorithm>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace impute {

class CLIParser {
public:
  CLIParser(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg.substr(0, 2) == "--") {
        std::string key = arg.substr(2);
        if (i + 1 < argc && argv[i + 1][0] != '-') {
          args[key] = argv[++i];
        } else {
          args[key] = "true";
        }
      }
    }
  }

  bool has(const std::string &key) const {
    return args.find(key) != args.end();
  }

  std::string get(const std::string &key,
                  const std::string &default_val = "") const {
    auto it = args.find(key);
    return (it != args.end()) ? it->second : default_val;
  }

  int get_int(const std::string &key, int default_val = 0) const {
    if (!has(key))
      return default_val;
    return std::stoi(args.at(key));
  }

  float get_float(const std::string &key, float default_val = 0.0f) const {
    if (!has(key))
      return default_val;
    return std::stof(args.at(key));
  }

  void print_help() const {
    std::cout << "Usage: impute [options]\n"
              << "Options:\n"
              << "  --algo <name>    Algorithm to run (rlsp, bgs, svd, bpca, "
                 "lls, ills, knn, sknn, iknn, ls, slls, gmc, cmve, amvi, arls, "
                 "lincmb)\n"
              << "  --input <path>   Input binary file path\n"
              << "  --mask <path>    Mask binary file path\n"
              << "  --eval <dir>     Evaluation data directory\n"
              << "  --k <int>        Number of neighbors (K)\n"
              << "  --pc <int>       Number of principal components (PC)\n"
              << "  --rank <int>     SVD Rank\n"
              << "  --ridge <float>  Ridge regularization factor\n"
              << "  --genes <int>    Max genes for BGS\n"
              << "  --iter <int>     Max iterations for EM algorithms\n"
              << "  --chunk <int>    Chunk size for streaming (default: 2048)\n"
              << "  --help           Show this help message\n";
  }

private:
  std::map<std::string, std::string> args;
};

} // namespace impute
