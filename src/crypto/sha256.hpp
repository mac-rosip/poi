#pragma once

#include <vector>
#include <array>
#include <cstdint>

namespace crypto {

std::array<uint8_t, 32> sha256(const uint8_t* data, size_t len);
std::array<uint8_t, 32> sha256(const std::vector<uint8_t>& data);
std::array<uint8_t, 32> double_sha256(const uint8_t* data, size_t len);
std::array<uint8_t, 32> double_sha256(const std::vector<uint8_t>& data);

} // namespace crypto