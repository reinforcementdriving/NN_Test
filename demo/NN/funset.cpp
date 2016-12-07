#include "funset.hpp"
#include <iostream>
#include <fstream>
#include <perceptron.hpp>
#include <BP.hpp>
#include <CNN.hpp>
#include <opencv2/opencv.hpp>

int test_perceptron()
{
	// prepare data
	const int len_data = 20;
	const int feature_dimension = 2;
	float data[len_data][feature_dimension] = {
		{ 10.3, 10.7 }, { 20.1, 100.8 }, { 44.9, 8.0 }, { -2.2, 15.3 }, { -33.3, 77.7 },
		{ -10.4, 111.1 }, { 99.3, -2.2 }, { 222.2, -5.5 }, { 10.1, 10.1 }, { 66.6, 30.2 },
		{ 0.1, 0.2 }, { 1.2, 0.03 }, { 0.5, 4.6 }, { -22.3, -11.1 }, { -88.9, -12.3 },
		{ -333.3, -444.4 }, { -111.2, 0.5 }, { -6.6, 2.9 }, { 3.3, -100.2 }, { 5.6, -88.8 } };
	int label_[len_data] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };

	std::vector<ANN::feature> set_feature;
	std::vector<ANN::label> set_label;

	for (int i = 0; i < len_data; i++) {
		ANN::feature feature_single;
		for (int j = 0; j < feature_dimension; j++) {
			feature_single.push_back(data[i][j]);
		}

		set_feature.push_back(feature_single);
		set_label.push_back(label_[i]);

		feature_single.resize(0);
	}

	// train
	int iterates = 1000;
	float learn_rate = 0.5;
	int size_weight = feature_dimension;
	float bias = 2.5;
	ANN::Perceptron perceptron(iterates, learn_rate, size_weight, bias);
	perceptron.getDataset(set_feature, set_label);
	bool flag = perceptron.train();
	if (flag) {
		std::cout << "data set is linearly separable" << std::endl;
	}
	else {
		std::cout << "data set is linearly inseparable" << std::endl;
		return -1;
	}

	// predict
	ANN::feature feature1;
	feature1.push_back(636.6);
	feature1.push_back(881.8);
	std::cout << "the correct result label is 1, " << "the real result label is: " << perceptron.predict(feature1) << std::endl;

	ANN::feature feature2;
	feature2.push_back(-26.32);
	feature2.push_back(-255.95);
	std::cout << "the correct result label is -1, " << "the real result label is: " << perceptron.predict(feature2) << std::endl;

	return 0;
}

int test_BP_train()
{
	ANN::BP bp1;
	bp1.init();
	bp1.train();

	return 0;
}

int test_BP_predict()
{
	ANN::BP bp2;
	bool flag = bp2.readModelFile("E:/GitCode/NN_Test/data/bp.model");
	if (!flag) {
		std::cout << "read bp model error" << std::endl;
		return -1;
	}

	int target[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	std::string path_images = "E:/GitCode/NN_Test/data/images/digit/handwriting_1/";

	int* data_image = new int[width_image_BP * height_image_BP];

	for (int i = 0; i < 10; i++) {
		char ch[15];
		sprintf(ch, "%d", i);
		std::string str;
		str = std::string(ch);
		str += ".jpg";
		str = path_images + str;

		cv::Mat mat = cv::imread(str, 2 | 4);
		if (!mat.data) {
			std::cout << "read image error" << std::endl;
			return -1;
		}

		if (mat.channels() == 3) {
			cv::cvtColor(mat, mat, cv::COLOR_BGR2GRAY);
		}

		if (mat.cols != width_image_BP || mat.rows != height_image_BP) {
			cv::resize(mat, mat, cv::Size(width_image_BP, height_image_BP));
		}

		memset(data_image, 0, sizeof(int) * (width_image_BP * height_image_BP));

		for (int h = 0; h < mat.rows; h++) {
			uchar* p = mat.ptr(h);
			for (int w = 0; w < mat.cols; w++) {
				if (p[w] > 128) {
					data_image[h* mat.cols + w] = 1;
				}
			}
		}

		int ret = bp2.predict(data_image, mat.cols, mat.rows);
		std::cout << "correct result: " << i << ",    actual result: " << ret << std::endl;
	}

	delete[] data_image;

	return 0;
}

int test_CNN_train()
{
	ANN::CNN cnn1;
	cnn1.init();
	cnn1.train();

	return 0;
}

int test_CNN_predict()
{
	ANN::CNN cnn2;
	bool flag = cnn2.readModelFile("E:/GitCode/NN_Test/data/cnn.model");
	if (!flag) {
		std::cout << "read cnn model error" << std::endl;
		return -1;
	}

	int width{ 32 }, height{ 32 };
	std::vector<int> target{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	std::string image_path{ "E:/GitCode/NN_Test/data/images/digit/handwriting_2/" };

	for (auto i : target) {
		std::string str = std::to_string(i);
		str += ".png";
		str = image_path + str;

		cv::Mat src = cv::imread(str, 0);
		if (src.data == nullptr) {
			fprintf(stderr, "read image error: %s\n", str.c_str());
			return -1;
		}

		cv::Mat tmp(src.rows, src.cols, CV_8UC1, cv::Scalar::all(255));
		cv::subtract(tmp, src, tmp);

		cv::resize(tmp, tmp, cv::Size(width, height));

		auto ret = cnn2.predict(tmp.data, width, height);

		fprintf(stdout, "the actual digit is: %d, correct digit is: %d\n", ret, i);
	}

	return 0;
}

int test_compare_file()
{
	std::ifstream infile1;
	infile1.open("E:/GitCode/NN_Test/data/10_delta_output_.bin", std::ios::in | std::ios::binary);
	if (!infile1.is_open()) {
		fprintf(stderr, "failed to open file\n");
		return -1;
	}

	std::ifstream infile2;
	infile2.open("E:/GitCode/NN_Test/data/10_delta_output.bin", std::ios::in | std::ios::binary);
	if (!infile2.is_open()) {
		fprintf(stderr, "failed to open file\n");
		return -1;
	}

	size_t length1 = 0, length2 = 0;

	infile1.read((char*)&length1, sizeof(size_t));
	infile2.read((char*)&length2, sizeof(size_t));

	if (length1 != length2) {
		fprintf(stderr, "their length is mismatch: required length: %d, actual length: %d\n", length1, length2);
		return -1;
	}

	double* data1 = new double[length1];
	double* data2 = new double[length2];

	for (int i = 0; i < length1; i++) {
		infile1.read((char*)&data1[i], sizeof(double));
		infile2.read((char*)&data2[i], sizeof(double));

		if (data1[i] != data2[i]) {
			fprintf(stderr, "no equal: %d: %f, %f\n", i, data1[i], data2[i]);
		}
	}

	delete[] data1;
	delete[] data2;

	infile1.close();
	infile2.close();

}
