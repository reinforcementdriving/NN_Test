#include "opencv.hpp"
#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include "common.hpp"


////////////////////////////////// Logistic Regression ///////////////////////////////
static void show_image(const cv::Mat& data, int columns, const std::string& name)
{
	cv::Mat big_image;
	for (int i = 0; i < data.rows; ++i) {
		big_image.push_back(data.row(i).reshape(0, columns));
	}

	cv::imshow(name, big_image);
	cv::waitKey(0);
}

static float calculate_accuracy_percent(const cv::Mat& original, const cv::Mat& predicted)
{
	return 100 * (float)cv::countNonZero(original == predicted) / predicted.rows;
}

int test_opencv_logistic_regression_train()
{
	const std::string image_path{ "E:/GitCode/NN_Test/data/images/digit/handwriting_0_and_1/" };
	cv::Mat data, labels, result;

	for (int i = 1; i < 11; ++i) {
		const std::vector<std::string> label{ "0_", "1_" };

		for (const auto& value : label) {
			std::string name = std::to_string(i);
			name = image_path + value + name + ".jpg";

			cv::Mat image = cv::imread(name, 0);
			if (image.empty()) {
				fprintf(stderr, "read image fail: %s\n", name.c_str());
				return -1;
			}

			data.push_back(image.reshape(0, 1));
		}
	}

	data.convertTo(data, CV_32F);
	//show_image(data, 28, "train data");

	std::unique_ptr<float[]> tmp(new float[20]);
	for (int i = 0; i < 20; ++i) {
		if (i % 2 == 0) tmp[i] = 0.f;
		else tmp[i] = 1.f;
	}
	labels = cv::Mat(20, 1, CV_32FC1, tmp.get());

	cv::Ptr<cv::ml::LogisticRegression> lr = cv::ml::LogisticRegression::create();
	lr->setLearningRate(0.00001);
	lr->setIterations(100);
	lr->setRegularization(cv::ml::LogisticRegression::REG_DISABLE);
	lr->setTrainMethod(cv::ml::LogisticRegression::MINI_BATCH);
	lr->setMiniBatchSize(1);

	CHECK(lr->train(data, cv::ml::ROW_SAMPLE, labels));

	const std::string save_file{ "E:/GitCode/NN_Test/data/logistic_regression_model.xml" }; // .xml, .yaml, .jsons
	lr->save(save_file);

	return 0;
}

int test_opencv_logistic_regression_predict()
{
	const std::string image_path{ "E:/GitCode/NN_Test/data/images/digit/handwriting_0_and_1/" };
	cv::Mat data, labels, result;

	for (int i = 11; i < 21; ++i) {
		const std::vector<std::string> label{ "0_", "1_" };

		for (const auto& value : label) {
			std::string name = std::to_string(i);
			name = image_path + value + name + ".jpg";

			cv::Mat image = cv::imread(name, 0);
			if (image.empty()) {
				fprintf(stderr, "read image fail: %s\n", name.c_str());
				return -1;
			}

			data.push_back(image.reshape(0, 1));
		}
	}

	data.convertTo(data, CV_32F);
	//show_image(data, 28, "test data");

	std::unique_ptr<int[]> tmp(new int[20]);
	for (int i = 0; i < 20; ++i) {
		if (i % 2 == 0) tmp[i] = 0;
		else tmp[i] = 1;
	}
	labels = cv::Mat(20, 1, CV_32SC1, tmp.get());

	const std::string model_file{ "E:/GitCode/NN_Test/data/logistic_regression_model.xml" };
	cv::Ptr<cv::ml::LogisticRegression> lr = cv::ml::LogisticRegression::load(model_file);

	lr->predict(data, result);

	fprintf(stdout, "predict result: \n");
	std::cout << "actual: " << labels.t() << std::endl;
	std::cout << "target: " << result.t() << std::endl;
	fprintf(stdout, "accuracy: %.2f%%\n", calculate_accuracy_percent(labels, result));

	return 0;
}
