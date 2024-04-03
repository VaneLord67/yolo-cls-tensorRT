#include "classifier.h"

py::array_t<uint8_t> mat2nparray(cv::Mat& mat) {
    // ����NumPy�������״
    std::vector<ptrdiff_t> numpy_shape;
    for (int i = 0; i < mat.dims; ++i) {
        numpy_shape.push_back(mat.size[i]);
    }
    // ����NumPy����
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
    // ���� TensorRT logger
    Logger logger;

    // ���ļ��ж�ȡ�ѱ��������
    std::ifstream engineFile(config.m_model_file_path, std::ios::binary);
    std::vector<char> engineData((std::istreambuf_iterator<char>(engineFile)), std::istreambuf_iterator<char>());
    engineFile.close();

    // ʹ�ö�ȡ�����ݴ�������ʱ������
    m_runtime = createInferRuntime(logger);
    m_engine = m_runtime->deserializeCudaEngine(engineData.data(), engineData.size(), nullptr);

    // ����ִ��������
    m_context = m_engine->createExecutionContext();

	cudaMalloc(&m_deviceInput, sizeof(float) * 224 * 224 * 3); // ���������� 224x224x3
	cudaMalloc(&m_deviceOutput, sizeof(float) * 1000); // ��������� 1000 ��

    m_outputData = new float[1000];
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
    // ִ������
    void* bindings[] = { m_deviceInput, m_deviceOutput };
    m_context->executeV2(bindings);
    // ����������ݻ�����
    cudaMemcpy(m_outputData, m_deviceOutput, sizeof(float) * 1000, cudaMemcpyDeviceToHost);
    return postprocess(m_outputData);
}
