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
    // ���� TensorRT logger
    Logger logger;

    // ���ļ��ж�ȡ�ѱ��������
    std::ifstream engineFile("E:/GraduationDesign/yolov8n-cls.trt", std::ios::binary);
    std::vector<char> engineData((std::istreambuf_iterator<char>(engineFile)), std::istreambuf_iterator<char>());
    engineFile.close();

    // ʹ�ö�ȡ�����ݴ�������ʱ������
    IRuntime* runtime = createInferRuntime(logger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size(), nullptr);

    // ����ִ��������
    IExecutionContext* context = engine->createExecutionContext();

    // ʹ�� OpenCV ��ȡͼƬ
    cv::Mat inputImage = cv::imread("E:/rtsp/dog.jpg");
    if (inputImage.empty()) {
        std::cerr << "Failed to read image!" << std::endl;
        return 1;
    }

    // �����豸�ڴ�
    void* deviceInput;
    void* deviceOutput;
    cudaMalloc(&deviceInput, sizeof(float) * 224 * 224 * 3); // ���������� 224x224x3
    cudaMalloc(&deviceOutput, sizeof(float) * 1000); // ��������� 1000 ��

    // ����Ԥ���������������һ����Ϊ preprocess �ĺ����������������ݣ�
    preprocess(inputImage, deviceInput);

    // ִ������
    void* bindings[] = { deviceInput, deviceOutput };
    context->executeV2(bindings);

    // ����������ݻ�����
    float* outputData = new float[1000];
    cudaMemcpy(outputData, deviceOutput, sizeof(float) * 1000, cudaMemcpyDeviceToHost);

    int maxIndex = postprocess(outputData);

    // �ͷ���Դ
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}

void preprocess(const cv::Mat& image, void* deviceInput) {
    // ����ͼ���СΪ224x224
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(224, 224));

    // ��ͨ��˳���BGRת��ΪRGB
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    // ��ͼ�����ݸ��Ƶ�CUDA�豸�ڴ���
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