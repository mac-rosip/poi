#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace chain {

std::string base58_encode(const std::vector<uint8_t>& data);
std::vector<uint8_t> base58_decode(const std::string& str);
std::string base58_encode_check(const std::vector<uint8_t>& data);
std::vector<uint8_t> base58_decode_check(const std::string& str);

} // namespace chain