// =============================================================================
// main.cpp — Hyperfanity standalone CLI entry point
// =============================================================================
//
// Usage:
//   hyperfanity --chain <trx|eth|sol> --prefix <pattern> [options]
//
// Options:
//   --chain <trx|eth|sol>   Target blockchain (required unless --benchmark)
//   --prefix <pattern>      Vanity prefix to match
//   --suffix <pattern>      Vanity suffix to match
//   --contains <pattern>    Match pattern anywhere in address
//   --benchmark             Run in benchmark mode (no pattern matching)
//   --devices <0,1,2>       Comma-separated GPU device IDs (default: all)
//   --iterations <n>        Iterations per kernel launch (default: 256)
//   --min-score <n>         Minimum score to report (default: pattern length)
//
// Examples:
//   hyperfanity --chain eth --prefix dead
//   hyperfanity --chain trx --suffix 8888
//   hyperfanity --chain sol --prefix Abc
//   hyperfanity --benchmark --chain eth
//
// =============================================================================

#include "arg_parser.hpp"
#include "types.hpp"
#include "hex_utils.hpp"
#include "dispatch/dispatcher.hpp"
#include "scoring/scorer.hpp"
#include "chain/chain.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <csignal>
#include <cstdlib>
#include <algorithm>
#include <sstream>

// Global dispatcher pointer for signal handling
static Dispatcher* g_dispatcher = nullptr;

static void signal_handler(int sig) {
    (void)sig;
    std::cerr << "\n[!] Interrupt received, stopping...\n";
    if (g_dispatcher) {
        g_dispatcher->stop();
    }
}

static void print_banner() {
    std::cout << R"(
  _   _                       __             _ _
 | | | |_   _ _ __   ___ _ _ / _| __ _ _ __ (_) |_ _   _
 | |_| | | | | '_ \ / _ \ '__| |_ / _` | '_ \| | __| | | |
 |  _  | |_| | |_) |  __/ |  |  _| (_| | | | | | |_| |_| |
 |_| |_|\__, | .__/ \___|_|  |_|  \__,_|_| |_|_|\__|\__, |
        |___/|_|                                      |___/
)" << std::endl;
    std::cout << "  CUDA Vanity Address Generator — TRX / ETH / SOL\n" << std::endl;
}

static void print_usage() {
    std::cout << "Usage: hyperfanity --chain <trx|eth|sol> --prefix <pattern> [options]\n\n"
              << "Options:\n"
              << "  --chain <trx|eth|sol>   Target blockchain\n"
              << "  --prefix <pattern>      Vanity prefix to match\n"
              << "  --suffix <pattern>      Vanity suffix to match\n"
              << "  --contains <pattern>    Match pattern anywhere\n"
              << "  --benchmark             Benchmark mode (no matching)\n"
              << "  --devices <0,1,2>       GPU device IDs (default: all)\n"
              << "  --iterations <n>        Iterations per launch (default: 256)\n"
              << "  --min-score <n>         Minimum score to report\n"
              << std::endl;
}

static ChainType parse_chain(const std::string& s) {
    std::string lower = s;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (lower == "trx" || lower == "tron")     return ChainType::TRON;
    if (lower == "eth" || lower == "ethereum") return ChainType::ETHEREUM;
    if (lower == "sol" || lower == "solana")   return ChainType::SOLANA;
    throw std::runtime_error("Unknown chain: " + s + " (use trx, eth, or sol)");
}

static std::vector<int> parse_devices(const std::string& s) {
    std::vector<int> ids;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        ids.push_back(std::stoi(token));
    }
    return ids;
}

static const char* chain_name(ChainType chain) {
    switch (chain) {
        case ChainType::TRON:     return "TRX";
        case ChainType::ETHEREUM: return "ETH";
        case ChainType::SOLANA:   return "SOL";
        default:                  return "???";
    }
}

int main(int argc, char* argv[]) {
    print_banner();

    if (argc < 2) {
        print_usage();
        return 1;
    }

    try {
        ArgParser args(argc, argv);

        // Check for help
        if (args.has_option("--help") || args.has_option("-h")) {
            print_usage();
            return 0;
        }

        bool benchmark = args.has_option("--benchmark");

        // Parse chain
        ChainType chain = ChainType::ETHEREUM;
        if (args.has_option("--chain")) {
            chain = parse_chain(args.get_option("--chain"));
        } else if (!benchmark) {
            std::cerr << "[!] Error: --chain is required (use trx, eth, or sol)\n";
            return 1;
        }

        // Parse pattern and mode
        ScoringConfig scoring_config;
        scoring_config.chain = chain;

        if (benchmark) {
            scoring_config.mode = ScoringMode::BENCHMARK;
            scoring_config.pattern = "";
        } else if (args.has_option("--prefix")) {
            scoring_config.mode = ScoringMode::PREFIX;
            scoring_config.pattern = args.get_option("--prefix");
        } else if (args.has_option("--suffix")) {
            scoring_config.mode = ScoringMode::SUFFIX;
            scoring_config.pattern = args.get_option("--suffix");
        } else if (args.has_option("--contains")) {
            scoring_config.mode = ScoringMode::ANYWHERE;
            scoring_config.pattern = args.get_option("--contains");
        } else {
            std::cerr << "[!] Error: specify --prefix, --suffix, --contains, or --benchmark\n";
            return 1;
        }

        // Case sensitivity: ETH hex is case-insensitive, TRX/SOL Base58 is case-sensitive
        scoring_config.case_sensitive = (chain != ChainType::ETHEREUM);

        // Min score
        if (args.has_option("--min-score")) {
            scoring_config.min_score = std::stoul(args.get_option("--min-score"));
        } else {
            scoring_config.min_score = scoring_config.pattern.empty()
                ? 1 : static_cast<uint32_t>(scoring_config.pattern.size());
        }

        Scorer scorer(scoring_config);

        // Build dispatcher config
        DispatcherConfig dispatch_config;
        dispatch_config.chain = chain;
        dispatch_config.curve = (chain == ChainType::SOLANA) ? CurveType::ED25519 : CurveType::SECP256K1;
        dispatch_config.scorer = scorer;
        dispatch_config.benchmark_mode = benchmark;

        if (args.has_option("--devices")) {
            dispatch_config.device_ids = parse_devices(args.get_option("--devices"));
        }

        if (args.has_option("--iterations")) {
            dispatch_config.iterations_per_launch = std::stoul(args.get_option("--iterations"));
        }

        // Print configuration
        std::cout << "  Chain:       " << chain_name(chain) << "\n";
        std::cout << "  Mode:        " << scoring_mode_name(scoring_config.mode) << "\n";
        if (!scoring_config.pattern.empty()) {
            std::cout << "  Pattern:     " << scoring_config.pattern << "\n";
        }
        std::cout << "  Min score:   " << scoring_config.min_score << "\n";
        std::cout << "  Iterations:  " << dispatch_config.iterations_per_launch << "\n";

        // Enumerate GPUs
        auto gpu_infos = GPUDevice::enumerate();
        if (gpu_infos.empty()) {
            std::cerr << "[!] Error: no CUDA devices found\n";
            return 1;
        }
        std::cout << "  GPUs:        " << gpu_infos.size() << " device(s)\n";
        for (const auto& info : gpu_infos) {
            bool selected = dispatch_config.device_ids.empty() ||
                std::find(dispatch_config.device_ids.begin(),
                          dispatch_config.device_ids.end(), info.id)
                    != dispatch_config.device_ids.end();
            std::cout << "    [" << info.id << "] " << info.name
                      << " (" << (info.total_memory / (1024*1024)) << " MB)"
                      << (selected ? " *" : "") << "\n";
        }
        std::cout << std::endl;

        // Create and initialize dispatcher
        Dispatcher dispatcher(dispatch_config);
        g_dispatcher = &dispatcher;

        // Install signal handler
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        std::cout << "[*] Initializing..." << std::flush;
        dispatcher.init();
        std::cout << " done.\n";

        std::cout << "[*] Mining started. Press Ctrl+C to stop.\n\n";

        // Progress callback
        auto progress = [&](double speed, uint64_t total, uint32_t best_score) {
            // Format speed
            std::string speed_str;
            if (speed >= 1e9) {
                char buf[64];
                snprintf(buf, sizeof(buf), "%.2f GH/s", speed / 1e9);
                speed_str = buf;
            } else if (speed >= 1e6) {
                char buf[64];
                snprintf(buf, sizeof(buf), "%.2f MH/s", speed / 1e6);
                speed_str = buf;
            } else if (speed >= 1e3) {
                char buf[64];
                snprintf(buf, sizeof(buf), "%.2f KH/s", speed / 1e3);
                speed_str = buf;
            } else {
                char buf[64];
                snprintf(buf, sizeof(buf), "%.0f H/s", speed);
                speed_str = buf;
            }

            // Format total
            std::string total_str;
            if (total >= 1000000000ULL) {
                char buf[64];
                snprintf(buf, sizeof(buf), "%.2fB", total / 1e9);
                total_str = buf;
            } else if (total >= 1000000ULL) {
                char buf[64];
                snprintf(buf, sizeof(buf), "%.2fM", total / 1e6);
                total_str = buf;
            } else {
                total_str = std::to_string(total);
            }

            std::cout << "\r  Speed: " << speed_str
                      << " | Total: " << total_str
                      << " | Best: " << best_score
                      << "    " << std::flush;
        };

        // Run mining loop
        VanityResult result = dispatcher.run(progress);

        std::cout << "\n\n";

        if (result.found) {
            std::cout << "========================================\n";
            std::cout << "  MATCH FOUND!\n";
            std::cout << "========================================\n";
            std::cout << "  Chain:       " << chain_name(result.chain) << "\n";
            std::cout << "  Address:     " << result.address << "\n";
            std::cout << "  Score:       " << result.score << "\n";
            std::cout << "  Private Key: " << toHex(result.private_key.data(),
                                                     result.private_key.size()) << "\n";
            std::cout << "========================================\n";
        } else {
            std::cout << "[*] Mining stopped. No match found.\n";
            std::cout << "  Total checked: " << dispatcher.total_checked() << "\n";
        }

        g_dispatcher = nullptr;
        return result.found ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "[!] Error: " << e.what() << "\n";
        return 1;
    }
}
