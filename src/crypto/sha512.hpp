#pragma once

#include <vector>
#include <array>
#include <cstdint>
#include <cstddef>

namespace crypto {

std::array<uint8_t, 64> sha512(const uint8_t* data, size_t len);
std::array<uint8_t, 64> sha512(const std::vector<uint8_t>& data);

} // namespace crypto