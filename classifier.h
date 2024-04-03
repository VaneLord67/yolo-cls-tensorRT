#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <logger.h>
#include <fstream>
#include <iostream>

namespace py = pybind11;
using namespace nvinfer1;
using namespace sample;

class ClassifierConfig {
public:
	std::string m_model_file_path;

	ClassifierConfig() : m_model_file_path() {};
};

class Classifier {

public:
	ClassifierConfig m_config;
	IExecutionContext* m_context;
	ICudaEngine* m_engine;
	IRuntime* m_runtime;
	void* m_deviceInput;
	void* m_deviceOutput;
	float* m_outputData;

	Classifier(ClassifierConfig config);

	~Classifier() {
		m_context->destroy();
		m_engine->destroy();
		m_runtime->destroy();
		if (m_outputData) {
			delete[] m_outputData;
		}
	}

	std::tuple<int, float> inference(py::array_t<uint8_t> input);
};