#pragma once
#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>


namespace impute {

enum class LogLevel { DEBUG, INFO, WARNING, ERROR };

class Logger {
public:
  static Logger &instance() {
    static Logger logger;
    return logger;
  }

  void set_level(LogLevel level) {
    std::lock_guard<std::mutex> lock(mutex_);
    min_level_ = level;
  }

  void log(LogLevel level, const std::string &message) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (level < min_level_)
      return;

    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::cout << "["
              << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S")
              << "] ";

    switch (level) {
    case LogLevel::DEBUG:
      std::cout << "\033[36m[DEBUG]\033[0m ";
      break;
    case LogLevel::INFO:
      std::cout << "\033[32m[INFO] \033[0m ";
      break;
    case LogLevel::WARNING:
      std::cout << "\033[33m[WARN] \033[0m ";
      break;
    case LogLevel::ERROR:
      std::cout << "\033[31m[ERROR]\033[0m ";
      break;
    }

    std::cout << message << std::endl;
  }

  static void debug(const std::string &msg) {
    instance().log(LogLevel::DEBUG, msg);
  }
  static void info(const std::string &msg) {
    instance().log(LogLevel::INFO, msg);
  }
  static void warn(const std::string &msg) {
    instance().log(LogLevel::WARNING, msg);
  }
  static void error(const std::string &msg) {
    instance().log(LogLevel::ERROR, msg);
  }

private:
  Logger() : min_level_(LogLevel::INFO) {}
  LogLevel min_level_;
  std::mutex mutex_;
};

} // namespace impute
