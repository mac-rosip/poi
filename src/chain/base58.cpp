#include "base58.hpp"
#include "../crypto/sha256.hpp"
#include <algorithm>
#include <stdexcept>

namespace chain {

const std::string BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

std::string base58_encode(const std::vector<uint8_t>& data) {
    if (data.empty()) return "";

    // Count leading zeros
    size_t leading_zeros = 0;
    for (auto byte : data) {
        if (byte == 0) ++leading_zeros;
        else break;
    }

    // Convert to big-endian number
    std::vector<uint8_t> temp = data;
    std::string result;

    while (!temp.empty() && !(temp.size() == 1 && temp[0] == 0)) {
        // Divide by 58
        uint32_t remainder = 0;
        for (size_t i = 0; i < temp.size(); ++i) {
            uint32_t value = (remainder << 8) | temp[i];
            temp[i] = value / 58;
            remainder = value % 58;
        }

        result.push_back(BASE58_ALPHABET[remainder]);

        // Remove leading zeros
        while (!temp.empty() && temp[0] == 0) {
            temp.erase(temp.begin());
        }
    }

    // Add leading '1's for leading zeros
    result.append(leading_zeros, '1');
    std::reverse(result.begin(), result.end());

    return result;
}

std::vector<uint8_t> base58_decode(const std::string& str) {
    if (str.empty()) return {};

    // Count leading '1's
    size_t leading_ones = 0;
    for (char c : str) {
        if (c == '1') ++leading_ones;
        else break;
    }

    std::vector<uint8_t> result;
    for (char c : str) {
        if (c == '1' && result.empty()) continue; // Skip leading '1's after counting

        auto pos = BASE58_ALPHABET.find(c);
        if (pos == std::string::npos) {
            throw std::invalid_argument("Invalid base58 character");
        }

        // Multiply result by 58 and add pos
        uint32_t carry = pos;
        for (size_t i = 0; i < result.size(); ++i) {
            uint32_t value = (uint32_t)result[i] * 58 + carry;
            result[i] = value & 0xFF;
            carry = value >> 8;
        }
        while (carry) {
            result.push_back(carry & 0xFF);
            carry >>= 8;
        }
    }

    // Add leading zeros
    result.insert(result.begin(), leading_ones, 0);
    std::reverse(result.begin(), result.end());

    return result;
}

std::string base58_encode_check(const std::vector<uint8_t>& data) {
    auto hash = crypto::double_sha256(data.data(), data.size());
    std::vector<uint8_t> with_checksum = data;
    with_checksum.insert(with_checksum.end(), hash.begin(), hash.begin() + 4);
    return base58_encode(with_checksum);
}

std::vector<uint8_t> base58_decode_check(const std::string& str) {
    auto decoded = base58_decode(str);
    if (decoded.size() < 4) {
        throw std::invalid_argument("Invalid base58 check string: too short");
    }

    std::vector<uint8_t> data(decoded.begin(), decoded.end() - 4);
    auto hash = crypto::double_sha256(data.data(), data.size());
    if (!std::equal(hash.begin(), hash.begin() + 4, decoded.end() - 4)) {
        throw std::invalid_argument("Invalid base58 check string: checksum mismatch");
    }

    return data;
}

} // namespace chain