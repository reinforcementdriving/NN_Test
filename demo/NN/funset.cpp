#include "funset.hpp"
#include <iostream>
#include "perceptron.hpp"
#include "BP.hpp""
#include "CNN.hpp"
#include "linear_regression.hpp"
#include "naive_bayes_classifier.hpp"
#include "common.hpp"
#include <opencv2/opencv.hpp>

// ================================ naive bayes classifier =====================
int test_naive_bayes_classifier_train()
{
	std::vector<ANN::sex_info<float>> info;
	info.push_back({ 6.f, 180.f, 12.f, 1 });
	info.push_back({5.92f, 190.f, 11.f, 1});
	info.push_back({5.58f, 170.f, 12.f, 1});
	info.push_back({5.92f, 165.f, 10.f, 1});
	info.push_back({ 5.f, 100.f, 6.f, 0 });
	info.push_back({5.5f, 150.f, 8.f, 0});
	info.push_back({5.42f, 130.f, 7.f, 0});
	info.push_back({5.75f, 150.f, 9.f, 0});

	ANN::NaiveBayesClassifier<float> naive_bayes;
	int ret = naive_bayes.init(info);
	if (ret != 0) {
		fprintf(stderr, "naive bayes classifier init fail: %d\n", ret);
		return -1;
	}

	const std::string model{ "E:/GitCode/NN_Test/data/naive_bayes_classifier.model" };
	ret = naive_bayes.train(model);
	if (ret != 0) {
		fprintf(stderr, "naive bayes classifier train fail: %d\n", ret);
		return -1;
	}

	return 0;
}

int test_naive_bayes_classifier_predict()
{
	ANN::sex_info<float> info = { 6.0f, 130.f, 8.f, -1 };

	ANN::NaiveBayesClassifier<float> naive_bayes;
	const std::string model{ "E:/GitCode/NN_Test/data/naive_bayes_classifier.model" };
	int ret = naive_bayes.load_model(model);
	if (ret != 0) {
		fprintf(stderr, "load naive bayes classifier model fail: %d\n", ret);
		return -1;
	}

	ret = naive_bayes.predict(info);
	if (ret == 0) fprintf(stdout, "It is a female\n");
	else fprintf(stdout, "It is a male\n");

	return 0;
}

// ============================ linear regression =============================
int test_linear_regression_train()
{
	std::vector<float> x{6.2f, 9.5f, 10.5f, 7.7f, 8.6f, 6.9f, 7.3f, 2.2f, 5.7f, 2.f,
		2.5f, 4.f, 5.4f, 2.2f, 7.2f, 12.2f, 5.6f, 9.f, 3.6f, 5.f,
		11.3f, 3.4f, 11.9f, 10.5f, 10.7f, 10.8f, 4.8f};
	std::vector<float> y{ 29.f, 44.f, 36.f, 37.f, 53.f, 18.f, 31.f, 14.f, 11.f, 11.f,
		22.f, 16.f, 27.f, 9.f, 29.f, 46.f, 23.f, 39.f, 15.f, 32.f,
		34.f, 17.f, 46.f, 42.f, 43.f, 34.f, 19.f };
	CHECK(x.size() == y.size());

	ANN::LinearRegression<float> lr;

	lr.set_regression_method(ANN::GRADIENT_DESCENT);
	lr.init(x.data(), y.data(), x.size());

	float learning_rate{ 0.001f };
	int iterations{ 1000 };
	std::string model{ "E:/GitCode/NN_Test/data/linear_regression.model" };

	int ret = lr.train(model, learning_rate, iterations);
	if (ret != 0) {
		fprintf(stderr, "train fail\n");
		return -1;
	}

	std::cout << lr << std::endl; // y = wx + b

	return 0;
}
int test_linear_regression_predict()
{
	ANN::LinearRegression<float> lr;

	std::string model{ "E:/GitCode/NN_Test/data/linear_regression.model" };
	int ret = lr.load_model(model);
	if (ret != 0) {
		fprintf(stderr, "load model fail: %s\n", model.c_str());
		return -1;
	}

	float x = 13.8f;
	float result = lr.predict(x);
	fprintf(stdout, "input value: %f, result value: %f\n", x, result);

	return 0;
}

// =============================== perceptron =================================
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

// =================================== BP =====================================
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

// =================================== CNN ====================================
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

