#pragma once
#include <chrono>
#include <deque>
#include <string>
#include <cstdint>
#include <cstddef>

class SpeedSample {
public:
    SpeedSample(size_t windowSize = 20);

    // Record that `count` keys were checked
    void sample(uint64_t count);

    // Get current speed in keys/second
    double getSpeed() const;

    // Get formatted speed string like "123.45 MH/s"
    std::string getSpeedString() const;

    // Get total keys checked
    uint64_t getTotal() const;

private:
    struct Entry {
        std::chrono::steady_clock::time_point time;
        uint64_t count;
    };

    std::deque<Entry> m_samples;
    size_t m_windowSize;
    uint64_t m_total;
};
