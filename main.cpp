#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <torch/script.h>

#define TORCH_STREAM_LOADING 1

enum class Arguments : size_t {
    ExecutablePath  = 0,
    ModelPath,
    ImagePath,
    NumberOfArguments
};

void log(const std::string& message) {
    std::chrono::time_point n = std::chrono::system_clock::now();
    std::time_t now = std::chrono::system_clock::to_time_t(n);
    std::tm* tm = std::localtime(&now);
    std::cout << "[" << std::put_time(tm, "%Y-%m-%d %H:%M:%S") << "] " << message << std::endl;
}

/**
 * @brief
 * @param argc arguments count
 * @param argv arguments value: [0] self executable path, [1] model path, [2] image path
 */
int main(int argc, char** argv)
{
    std::vector<std::string> arguments;
    for (int i = 0; i < argc; ++i) {
        arguments.push_back(argv[i]);
    }

    for (const auto& v : arguments) {
        std::cout << v << std::endl;
    }

    try {
        log("begin model loading");
        c10::DeviceType d = c10::kCUDA;
        torch::jit::ExtraFilesMap extras;
#if TORCH_STREAM_LOADING
        std::ifstream s(arguments[static_cast<size_t>(Arguments::ModelPath)]);
        torch::jit::Module m = torch::jit::load(s, d, extras);
#else
        torch::jit::Module m = torch::jit::load(arguments[static_cast<size_t>(Arguments::ModelPath)], d, extras);
#endif
        log("end model loading");

        log("model begin to prediction mode");
        m.eval();
        log("model end to prediction mode");
    } catch (const c10::Error& e) {
        log(e.msg());
    } catch (const std::exception& e) {
        log(e.what());
    }

    return 0;
}
