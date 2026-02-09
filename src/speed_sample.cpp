#include "speed_sample.hpp"
#include <cmath>
#include <sstream>
#include <iomanip>

SpeedSample::SpeedSample(size_t windowSize)
    : m_windowSize(windowSize), m_total(0) {}

void SpeedSample::sample(uint64_t count) {
    m_total += count;
    m_samples.push_back({
        std::chrono::steady_clock::now(),
        count
    });

    while (m_samples.size() > m_windowSize) {
        m_samples.pop_front();
    }
}

double SpeedSample::getSpeed() const {
    if (m_samples.size() < 2) {
        return 0.0;
    }

    uint64_t totalKeys = 0;
    for (const auto& entry : m_samples) {
        totalKeys += entry.count;
    }

    auto timeDiff = m_samples.back().time - m_samples.front().time;
    double seconds = std::chrono::duration<double>(timeDiff).count();

    if (seconds < 0.001) {
        return 0.0;
    }

    return static_cast<double>(totalKeys) / seconds;
}

std::string SpeedSample::getSpeedString() const {
    double speed = getSpeed();

    const char* unit = "H/s";
    if (speed >= 1e9) {
        speed /= 1e9;
        unit = "GH/s";
    } else if (speed >= 1e6) {
        speed /= 1e6;
        unit = "MH/s";
    } else if (speed >= 1e3) {
        speed /= 1e3;
        unit = "KH/s";
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << speed << " " << unit;
    return oss.str();
}

uint64_t SpeedSample::getTotal() const {
    return m_total;
}
