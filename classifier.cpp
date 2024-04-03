#include "classifier.h"

py::array_t<uint8_t> mat2nparray(cv::Mat& mat) {
    // 创建NumPy数组的形状
    std::vector<ptrdiff_t> numpy_shape;
    for (int i = 0; i < mat.dims; ++i) {
        numpy_shape.push_back(mat.size[i]);
    }
    // 创建NumPy数组
    py::array_t<uint8_t> nparray(numpy_shape);
    return py::array_t<uint8_t>({ mat.rows,mat.cols,3 }, mat.data);
}

cv::Mat nparray2mat(py::array_t<uint8_t> nparray) {
    auto ptr = nparray.mutable_data();
    auto shape = nparray.shape();
    cv::Mat mat(shape[0], shape[1], CV_8UC3, ptr);
    return mat;
}

Classifier::Classifier(ClassifierConfig config) {
    // 创建 TensorRT logger
    Logger logger;

    // 从文件中读取已保存的引擎
    std::ifstream engineFile(config.m_model_file_path, std::ios::binary);
    std::vector<char> engineData((std::istreambuf_iterator<char>(engineFile)), std::istreambuf_iterator<char>());
    engineFile.close();

    // 使用读取的数据创建运行时和引擎
    m_runtime = createInferRuntime(logger);
    m_engine = m_runtime->deserializeCudaEngine(engineData.data(), engineData.size(), nullptr);

    // 创建执行上下文
    m_context = m_engine->createExecutionContext();

	cudaMalloc(&m_deviceInput, sizeof(float) * 224 * 224 * 3); // 假设输入是 224x224x3
	cudaMalloc(&m_deviceOutput, sizeof(float) * 1000); // 假设输出是 1000 类

    m_outputData = new float[1000];
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

std::tuple<int, float> postprocess(float* outputData) {
    float maxScore = 0;
    int maxIndex = -1;
    for (int i = 0; i < 1000; i++) {
        if (outputData[i] > maxScore) {
            maxScore = outputData[i];
            maxIndex = i;
        }
    }
    return std::make_tuple<>(maxIndex, maxScore);
}

std::tuple<int, float> Classifier::inference(py::array_t<uint8_t> input) {
    cv::Mat mat = nparray2mat(input);
    preprocess(mat, m_deviceInput);
    // 执行推理
    void* bindings[] = { m_deviceInput, m_deviceOutput };
    m_context->executeV2(bindings);
    // 拷贝输出数据回主机
    cudaMemcpy(m_outputData, m_deviceOutput, sizeof(float) * 1000, cudaMemcpyDeviceToHost);
    return postprocess(m_outputData);
}
