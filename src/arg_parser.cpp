#include "arg_parser.hpp"

ArgParser::ArgParser(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg.substr(0, 2) == "--") {
            // Long option
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                options_[arg] = argv[++i];
            } else {
                options_[arg] = "";
            }
        } else if (arg[0] == '-') {
            // Short option
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                options_[arg] = argv[++i];
            } else {
                options_[arg] = "";
            }
        } else {
            positional_args_.push_back(arg);
        }
    }
}

bool ArgParser::has_option(const std::string& option) const {
    return options_.find(option) != options_.end();
}

std::string ArgParser::get_option(const std::string& option) const {
    auto it = options_.find(option);
    if (it == options_.end()) {
        throw ArgParseError("Option not found: " + option);
    }
    return it->second;
}

std::string ArgParser::get_option(const std::string& option, const std::string& default_value) const {
    auto it = options_.find(option);
    if (it == options_.end()) {
        return default_value;
    }
    return it->second;
}

std::vector<std::string> ArgParser::get_positional_args() const {
    return positional_args_;
}
