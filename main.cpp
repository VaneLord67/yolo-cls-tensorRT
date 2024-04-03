#include <fstream>
#include <iostream>
#include <vector>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <logger.h>

using namespace nvinfer1;
using namespace sample;

void preprocess(const cv::Mat& image, void* deviceInput);
int postprocess(float* outputData);

int main() {
    // 创建 TensorRT logger
    Logger logger;

    // 从文件中读取已保存的引擎
    std::ifstream engineFile("E:/GraduationDesign/yolov8n-cls.trt", std::ios::binary);
    std::vector<char> engineData((std::istreambuf_iterator<char>(engineFile)), std::istreambuf_iterator<char>());
    engineFile.close();

    // 使用读取的数据创建运行时和引擎
    IRuntime* runtime = createInferRuntime(logger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size(), nullptr);

    // 创建执行上下文
    IExecutionContext* context = engine->createExecutionContext();

    // 使用 OpenCV 读取图片
    cv::Mat inputImage = cv::imread("E:/rtsp/dog.jpg");
    if (inputImage.empty()) {
        std::cerr << "Failed to read image!" << std::endl;
        return 1;
    }

    // 分配设备内存
    void* deviceInput;
    void* deviceOutput;
    cudaMalloc(&deviceInput, sizeof(float) * 224 * 224 * 3); // 假设输入是 224x224x3
    cudaMalloc(&deviceOutput, sizeof(float) * 1000); // 假设输出是 1000 类

    // 数据预处理（这里假设你有一个名为 preprocess 的函数来处理输入数据）
    preprocess(inputImage, deviceInput);

    // 执行推理
    void* bindings[] = { deviceInput, deviceOutput };
    context->executeV2(bindings);

    // 拷贝输出数据回主机
    float* outputData = new float[1000];
    cudaMemcpy(outputData, deviceOutput, sizeof(float) * 1000, cudaMemcpyDeviceToHost);

    int maxIndex = postprocess(outputData);

    // 释放资源
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}

void preprocess(const cv::Mat& image, void* deviceInput) {
    // 调整图像大小为224x224
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(224, 224));

    // 将通道顺序从BGR转换为RGB
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    // 将图像数据复制到CUDA设备内存中
    const size_t channels = 3;
    const size_t height = 224;
    const size_t width = 224;
    std::vector<float> inputData(channels * height * width);
    for (size_t c = 0; c < channels; ++c) {
        for (size_t h = 0; h < height; ++h) {
            for (size_t w = 0; w < width; ++w) {
                inputData[c * height * width + h * width + w] =
                    static_cast<float>(resized.at<cv::Vec3b>(h, w)[c]) / 255.0f;
            }
        }
    }
    cudaMemcpy(deviceInput, inputData.data(), sizeof(float) * channels * height * width, cudaMemcpyHostToDevice);
}

int postprocess(float* outputData) {
    float maxScore = 0;
    int maxIndex = -1;
    for (int i = 0; i < 1000; i++) {
        if (outputData[i] > maxScore) {
            maxScore = outputData[i];
            maxIndex = i;
        }
    }
    std::cout << "maxIndex = " << maxIndex << std::endl;
    std::cout << "maxScore = " << maxScore << std::endl;
    return maxIndex;
}