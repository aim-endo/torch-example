#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/script.h>
#include <torch/nn.h>

#include <opencv2/opencv.hpp>

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
        const c10::DeviceType d = c10::kCUDA;
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

        log("begin image loading");
        cv::Mat image = cv::imread(arguments[static_cast<size_t>(Arguments::ImagePath)]);
        if (image.cols != image.rows) {
            std::stringstream ss;
            ss << "image is not square:";
            ss << " cols = " << image.cols;
            ss << " rows = " << image.rows;
            throw std::runtime_error(ss.str());
        }
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        log("end image loading");

        // image -> tensor
        log("begin image to tensor");
        torch::TensorOptions input_options(torch::kUInt8);
        std::array<int64_t, 4> dims{
            1L,
            static_cast<int64_t>(image.rows),
            static_cast<int64_t>(image.cols),
            static_cast<int64_t>(image.channels())
        };
        torch::Tensor input_tensor = torch::from_blob(image.data, c10::ArrayRef<int64_t>(dims), input_options).to(c10::kCUDA); // image -> tensor
        input_tensor = input_tensor.transpose(2, 3).transpose(1, 2); // HWC -> CHW
        log("end image to tensor");

        // resize
        log("begin resize tensor");
        const size_t size = 256;
        input_tensor = input_tensor.to(c10::TensorOptions().dtype(c10::kFloat));
        const auto resize_options = torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({size, size})).mode(torch::kBilinear).align_corners(false);
        input_tensor = torch::nn::functional::interpolate(input_tensor, resize_options).clamp_min(0.0).clamp_max(255.0);
        log("end resize tensor");

        // normalize
        /* wip
        log("begin normalize tensor");
        torch::TensorOptions normalize_options(torch::kFloat32);
        log("end normalize tensor");
        */
    } catch (const c10::Error& e) {
        log(e.msg());
    } catch (const std::exception& e) {
        log(e.what());
    }

    return 0;
}
