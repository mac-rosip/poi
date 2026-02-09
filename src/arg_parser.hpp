#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>

class ArgParser {
public:
    ArgParser(int argc, char* argv[]);
    ~ArgParser() = default;

    bool has_option(const std::string& option) const;
    std::string get_option(const std::string& option) const;
    std::string get_option(const std::string& option, const std::string& default_value) const;
    std::vector<std::string> get_positional_args() const;

private:
    std::unordered_map<std::string, std::string> options_;
    std::vector<std::string> positional_args_;
};

class ArgParseError : public std::runtime_error {
public:
    ArgParseError(const std::string& msg) : std::runtime_error(msg) {}
};