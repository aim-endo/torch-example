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

enum class ResultType : size_t {
    CLASSIFICATION = 0,
    DETECTION,
    SEGMENTATION
};

typedef struct Input_ {
    cv::Mat     image;
    int64_t     size = 256;
    std::vector<std::string> labels = {"1", "2", "3", "4", "5", "6", "7", "8"};
    std::array<float, 3>       mean = {0.0, 0.0, 0.0};
    std::array<float, 3>       std = {1.0, 1.0, 1.0};
    ResultType  type = ResultType::DETECTION;
} Input;

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

    Input input;
    try {
        log("begin model loading");
        const c10::DeviceType device = c10::kCUDA;
        torch::jit::ExtraFilesMap extras;
#if TORCH_STREAM_LOADING
        std::ifstream s(arguments[static_cast<size_t>(Arguments::ModelPath)]);
        torch::jit::Module model = torch::jit::load(s, device, extras);
#else
        torch::jit::Module model = torch::jit::load(arguments[static_cast<size_t>(Arguments::ModelPath)], d, extras);
#endif
        log("end model loading");

        log("model begin to prediction mode");
        model.eval();
        log("model end to prediction mode");

        log("begin image loading");
        input.image = cv::imread(arguments[static_cast<size_t>(Arguments::ImagePath)]);
        if (input.image.cols != input.image.rows) {
            std::stringstream ss;
            ss << "image is not square:";
            ss << " cols = " << input.image.cols;
            ss << " rows = " << input.image.rows;
            throw std::runtime_error(ss.str());
        }
        cv::cvtColor(input.image, input.image, cv::COLOR_BGR2RGB);
        log("end image loading");

        // image -> tensor
        log("begin image to tensor");
        torch::TensorOptions input_options(torch::kUInt8);
        std::array<int64_t, 4> dims{
            1L,
            static_cast<int64_t>(input.image.rows),
            static_cast<int64_t>(input.image.cols),
            static_cast<int64_t>(input.image.channels())
        };
        torch::Tensor tensor = torch::from_blob(input.image.data, c10::ArrayRef<int64_t>(dims), input_options).to(device); // image -> tensor
        tensor = tensor.transpose(2, 3).transpose(1, 2); // HWC -> CHW
        log("end image to tensor");

        // resize
        log("begin resize tensor");
        tensor = tensor.to(c10::TensorOptions().dtype(c10::kFloat));
        const auto resize_options = torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({input.size, input.size})).mode(torch::kBilinear).align_corners(false);
        tensor = torch::nn::functional::interpolate(tensor, resize_options).clamp_min(0.0).clamp_max(255.0);
        log("end resize tensor");

        // normalize
        log("begin normalize tensor");
        torch::TensorOptions normalize_options(torch::kFloat32);
        const std::array<int64_t, 4> shape{1L, 3L, 1L, 1L};
        torch::Tensor mean = torch::from_blob(static_cast<void*>(input.mean.data()), c10::ArrayRef<int64_t>(shape), normalize_options);
        torch::Tensor std  = torch::from_blob(static_cast<void*>(input.std.data()), c10::ArrayRef<int64_t>(shape), normalize_options);
        torch::Tensor normalized = tensor.div(255.0).sub(mean).div(std).to(c10::TensorOptions().device(device));
        log("end normalize tensor");

        // forward (predict)
        torch::jit::IValue value = model.forward(std::vector<torch::jit::IValue>({normalized}));

        // postprocess
        if (input.type == ResultType::DETECTION) {
            const std::vector<c10::IValue> elements = value.toTuple()->elements();
            torch::Tensor classification = elements[0].toTensor().squeeze();
            torch::Tensor bboxes = elements[1].toTensor().squeeze();
        }
    } catch (const c10::Error& e) {
        log(e.msg());
    } catch (const std::exception& e) {
        log(e.what());
    }

    return 0;
}
