#include "funset.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>

static int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

static void read_Mnist(std::string filename, std::vector<cv::Mat> &vec)
{
	std::ifstream file(filename, std::ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);

		for (int i = 0; i < number_of_images; ++i) {
			cv::Mat tp = cv::Mat::zeros(n_rows, n_cols, CV_8UC1);
			for (int r = 0; r < n_rows; ++r) {
				for (int c = 0; c < n_cols; ++c) {
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					tp.at<uchar>(r, c) = (int)temp;
				}
			}
			vec.push_back(tp);
		}
	}
}

static void read_Mnist_Label(std::string filename, std::vector<int> &vec)
{
	std::ifstream file(filename, std::ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);

		for (int i = 0; i < number_of_images; ++i) {
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			vec[i] = (int)temp;
		}
	}
}

static std::string GetImageName(int number, int arr[])
{
	std::string str1, str2;

	for (int i = 0; i < 10; i++) {
		if (number == i) {
			arr[i]++;
			str1 = std::to_string(arr[i]);

			if (arr[i] < 10) {
				str1 = "0000" + str1;
			} else if (arr[i] < 100) {
				str1 = "000" + str1;
			} else if (arr[i] < 1000) {
				str1 = "00" + str1;
			} else if (arr[i] < 10000) {
				str1 = "0" + str1;
			}

			break;
		}
	}

	str2 = std::to_string(number) + "_" + str1;

	return str2;
}

int MNISTtoImage()
{
	// reference: http://eric-yuan.me/cpp-read-mnist/
	// test images and test labels
	// read MNIST image into OpenCV Mat vector
	std::string filename_test_images = "E:/GitCode/NN_Test/data/database/MNIST/t10k-images.idx3-ubyte";
	int number_of_test_images = 10000;
	std::vector<cv::Mat> vec_test_images;

	read_Mnist(filename_test_images, vec_test_images);

	// read MNIST label into int vector
	std::string filename_test_labels = "E:/GitCode/NN_Test/data/database/MNIST/t10k-labels.idx1-ubyte";
	std::vector<int> vec_test_labels(number_of_test_images);

	read_Mnist_Label(filename_test_labels, vec_test_labels);

	if (vec_test_images.size() != vec_test_labels.size()) {
		std::cout << "parse MNIST test file error" << std::endl;
		return -1;
	}

	// save test images
	int count_digits[10];
	std::fill(&count_digits[0], &count_digits[0] + 10, 0);

	std::string save_test_images_path = "E:/GitCode/NN_Test/data/tmp/MNIST/test_images/";

	for (int i = 0; i < vec_test_images.size(); i++) {
		int number = vec_test_labels[i];
		std::string image_name = GetImageName(number, count_digits);
		image_name = save_test_images_path + image_name + ".jpg";

		cv::imwrite(image_name, vec_test_images[i]);
	}

	// train images and train labels
	// read MNIST image into OpenCV Mat vector
	std::string filename_train_images = "E:/GitCode/NN_Test/data/database/MNIST/train-images.idx3-ubyte";
	int number_of_train_images = 60000;
	std::vector<cv::Mat> vec_train_images;

	read_Mnist(filename_train_images, vec_train_images);

	// read MNIST label into int vector
	std::string filename_train_labels = "E:/GitCode/NN_Test/data/database/MNIST/train-labels.idx1-ubyte";
	std::vector<int> vec_train_labels(number_of_train_images);

	read_Mnist_Label(filename_train_labels, vec_train_labels);

	if (vec_train_images.size() != vec_train_labels.size()) {
		std::cout << "parse MNIST train file error" << std::endl;
		return -1;
	}

	// save train images
	std::fill(&count_digits[0], &count_digits[0] + 10, 0);

	std::string save_train_images_path = "E:/GitCode/NN_Test/data/tmp/MNIST/train_images/";

	for (int i = 0; i < vec_train_images.size(); i++) {
		int number = vec_train_labels[i];
		std::string image_name = GetImageName(number, count_digits);
		image_name = save_train_images_path + image_name + ".jpg";

		cv::imwrite(image_name, vec_train_images[i]);
	}

	// save big imags
	std::string images_path = "E:/GitCode/NN_Test/data/tmp/MNIST/train_images/";
	int width = 28 * 20;
	int height = 28 * 10;
	cv::Mat dst(height, width, CV_8UC1);

	for (int i = 0; i < 10; i++) {
		for (int j = 1; j <= 20; j++) {
			int x = (j-1) * 28;
			int y = i * 28;
			cv::Mat part = dst(cv::Rect(x, y, 28, 28));

			std::string str = std::to_string(j);
			if (j < 10)
				str = "0000" + str;
			else
				str = "000" + str;

			str = std::to_string(i) + "_" + str + ".jpg";
			std::string input_image = images_path + str;

			cv::Mat src = cv::imread(input_image, 0);
			if (src.empty()) {
				fprintf(stderr, "read image error: %s\n", input_image.c_str());
				return -1;
			}

			src.copyTo(part);
		}
	}

	std::string output_image = images_path + "result.png";
	cv::imwrite(output_image, dst);

	return 0;
}

int CIFAR10toImage()
{
	return 0;
}

int CIFAR100toImage()
{
	return 0;
}
