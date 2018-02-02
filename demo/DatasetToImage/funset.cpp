#include "funset.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>

int ORLFacestoImage()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/79008891
	const std::string path{ "E:/GitCode/NN_Test/data/database/ORL_Faces/" };
	cv::Mat dst;
	int height, width;

	for (int i = 1; i <= 40; ++i) {
		std::string directory = path + "s" + std::to_string(i) + "/";

		for (int j = 1; j <= 10; ++j) {
			std::string image_name = directory + std::to_string(j) + ".pgm";
			cv::Mat mat = cv::imread(image_name, 0);
			if (!mat.data) {
				fprintf(stderr, "read image fail: %s\n", image_name.c_str());
			}

			//std::string save_image_name = directory + std::to_string(j) + ".png";
			//cv::imwrite(save_image_name, mat);

			if (i == 1 && j == 1) {
				height = mat.rows;
				width = mat.cols;
				dst = cv::Mat(height * 20, width * 20, CV_8UC1);
			}

			int y_start = (i - 1) / 2 * height;
			int y_end = y_start + height;
			int x_start = (i - 1) % 2 * 10 * width + (j - 1) * width;
			int x_end = x_start + width;
			cv::Mat copy = dst(cv::Range(y_start, y_end), cv::Range(x_start, x_end));
			mat.copyTo(copy);
		}
	}

	int new_width = 750;
	float factor = dst.cols * 1.f / new_width;
	int new_height = dst.rows / factor;
	cv::resize(dst, dst, cv::Size(new_width, new_height));
	cv::imwrite("E:/GitCode/NN_Test/data/orl_faces_dataset.png", dst);

	return 0;
}

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

		file.close();
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

		file.close();
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
	// Blog: http://blog.csdn.net/fengbingchun/article/details/49611549
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

static void write_image_cifar(const cv::Mat& bgr, const std::string& image_save_path, const std::vector<int>& label_count, int label_class)
{
	std::string str = std::to_string(label_count[label_class]);

	if (label_count[label_class] < 10) {
		str = "0000" + str;
	} else if (label_count[label_class] < 100) {
		str = "000" + str;
	} else if (label_count[label_class] < 1000) {
		str = "00" + str;
	} else if (label_count[label_class] < 10000) {
		str = "0" + str;
	} else {
		fprintf(stderr, "save image name fail\n");
		return;
	}

	str = std::to_string(label_class) + "_" + str + ".png";
	str = image_save_path + str;

	cv::imwrite(str, bgr);
}

static void read_cifar_10(const std::string& bin_name, const std::string& image_save_path, int image_count, std::vector<int>& label_count)
{
	int image_width = 32;
	int image_height = 32;

	std::ifstream file(bin_name, std::ios::binary);
	if (file.is_open()) {
		for (int i = 0; i < image_count; ++i) {
			cv::Mat red = cv::Mat::zeros(image_height, image_width, CV_8UC1);
			cv::Mat green = cv::Mat::zeros(image_height, image_width, CV_8UC1);
			cv::Mat blue = cv::Mat::zeros(image_height, image_width, CV_8UC1);

			int label_class = 0;
			file.read((char*)&label_class, 1);
			label_count[label_class]++;

			file.read((char*)red.data, 1024);
			file.read((char*)green.data, 1024);
			file.read((char*)blue.data, 1024);

			std::vector<cv::Mat> tmp{ blue, green, red };
			cv::Mat bgr;
			cv::merge(tmp, bgr);

			write_image_cifar(bgr, image_save_path, label_count, label_class);
		}

		file.close();
	}
}

int CIFAR10toImage()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/53560637
	std::string images_path = "E:/GitCode/NN_Test/data/database/CIFAR/CIFAR-10/";
	// train image
	std::vector<int> label_count(10, 0);
	for (int i = 1; i <= 5; i++) {
		std::string bin_name = images_path + "data_batch_" + std::to_string(i) + ".bin";
		std::string image_save_path = "E:/GitCode/NN_Test/data/tmp/cifar-10_train/";
		int image_count = 10000;

		read_cifar_10(bin_name, image_save_path, image_count, label_count);
	}

	// test image
	std::fill(&label_count[0], &label_count[0] + 10, 0);
	std::string bin_name = images_path + "test_batch.bin";
	std::string image_save_path = "E:/GitCode/NN_Test/data/tmp/cifar-10_test/";
	int image_count = 10000;

	read_cifar_10(bin_name, image_save_path, image_count, label_count);

	// save big imags
	images_path = "E:/GitCode/NN_Test/data/tmp/cifar-10_train/";
	int width = 32 * 20;
	int height = 32 * 10;
	cv::Mat dst(height, width, CV_8UC3);

	for (int i = 0; i < 10; i++) {
		for (int j = 1; j <= 20; j++) {
			int x = (j - 1) * 32;
			int y = i * 32;
			cv::Mat part = dst(cv::Rect(x, y, 32, 32));

			std::string str = std::to_string(j);
			if (j < 10)
				str = "0000" + str;
			else
				str = "000" + str;

			str = std::to_string(i) + "_" + str + ".png";
			std::string input_image = images_path + str;

			cv::Mat src = cv::imread(input_image, 1);
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

static void write_image_cifar(const cv::Mat& bgr, const std::string& image_save_path,
	const std::vector<std::vector<int>>& label_count, int label_class_coarse, int label_class_fine)
{
	std::string str = std::to_string(label_count[label_class_coarse][label_class_fine]);

	if (label_count[label_class_coarse][label_class_fine] < 10) {
		str = "0000" + str;
	} else if (label_count[label_class_coarse][label_class_fine] < 100) {
		str = "000" + str;
	} else if (label_count[label_class_coarse][label_class_fine] < 1000) {
		str = "00" + str;
	} else if (label_count[label_class_coarse][label_class_fine] < 10000) {
		str = "0" + str;
	} else {
		fprintf(stderr, "save image name fail\n");
		return;
	}

	str = std::to_string(label_class_coarse) + "_" + std::to_string(label_class_fine) + "_" + str + ".png";
	str = image_save_path + str;

	cv::imwrite(str, bgr);
}

static void read_cifar_100(const std::string& bin_name, const std::string& image_save_path, int image_count, std::vector<std::vector<int>>& label_count)
{
	int image_width = 32;
	int image_height = 32;

	std::ifstream file(bin_name, std::ios::binary);
	if (file.is_open()) {
		for (int i = 0; i < image_count; ++i) {
			cv::Mat red = cv::Mat::zeros(image_height, image_width, CV_8UC1);
			cv::Mat green = cv::Mat::zeros(image_height, image_width, CV_8UC1);
			cv::Mat blue = cv::Mat::zeros(image_height, image_width, CV_8UC1);

			int label_class_coarse = 0;
			file.read((char*)&label_class_coarse, 1);
			int label_class_fine = 0;
			file.read((char*)&label_class_fine, 1);
			label_count[label_class_coarse][label_class_fine]++;

			file.read((char*)red.data, 1024);
			file.read((char*)green.data, 1024);
			file.read((char*)blue.data, 1024);

			std::vector<cv::Mat> tmp{ blue, green, red };
			cv::Mat bgr;
			cv::merge(tmp, bgr);

			write_image_cifar(bgr, image_save_path, label_count, label_class_coarse, label_class_fine);
		}

		file.close();
	}
}

int CIFAR100toImage()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/53560637
	std::string images_path = "E:/GitCode/NN_Test/data/database/CIFAR/CIFAR-100/";
	// train image
	std::vector<std::vector<int>> label_count;
	label_count.resize(20);
	for (int i = 0; i < 20; i++) {
		label_count[i].resize(100);
		std::fill(&label_count[i][0], &label_count[i][0] + 100, 0);
	}

	std::string bin_name = images_path + "train.bin";
	std::string image_save_path = "E:/GitCode/NN_Test/data/tmp/cifar-100_train/";
	int image_count = 50000;

	read_cifar_100(bin_name, image_save_path, image_count, label_count);

	// test image
	for (int i = 0; i < 20; i++) {
		label_count[i].resize(100);
		std::fill(&label_count[i][0], &label_count[i][0] + 100, 0);
	}
	bin_name = images_path + "test.bin";
	image_save_path = "E:/GitCode/NN_Test/data/tmp/cifar-100_test/";
	image_count = 10000;

	read_cifar_100(bin_name, image_save_path, image_count, label_count);

	// save big imags
	images_path = "E:/GitCode/NN_Test/data/tmp/cifar-100_train/";
	int width = 32 * 20;
	int height = 32 * 100;
	cv::Mat dst(height, width, CV_8UC3);
	std::vector<std::string> image_names;

	for (int j = 0; j < 20; j++) {
		for (int i = 0; i < 100; i++) {
			std::string str1 = std::to_string(j);
			std::string str2 = std::to_string(i);
			std::string str = images_path + str1 + "_" + str2 + "_00001.png";
			cv::Mat src = cv::imread(str, 1);
			if (src.data) {
				for (int t = 1; t < 21; t++) {
					if (t < 10)
						str = "0000" + std::to_string(t);
					else
						str = "000" + std::to_string(t);

					str = images_path + str1 + "_" + str2 + "_" + str + ".png";
					image_names.push_back(str);
				}
			}
		}
	}

	for (int i = 0; i < 100; i++) {
		for (int j = 0; j < 20; j++) {
			int x = j * 32;
			int y = i * 32;
			cv::Mat part = dst(cv::Rect(x, y, 32, 32));
			cv::Mat src = cv::imread(image_names[i * 20 + j], 1);
			if (src.empty()) {
				fprintf(stderr, "read image fail: %s\n", image_names[i * 20 + j].c_str());
				return -1;
			}

			src.copyTo(part);
		}
	}

	std::string output_image = images_path + "result.png";
	cv::imwrite(output_image, dst);

	cv::Mat src = cv::imread(output_image, 1);
	if (src.empty()) {
		fprintf(stderr, "read result image fail: %s\n", output_image.c_str());
		return -1;
	}
	for (int i = 0; i < 4; i++) {
		cv::Mat dst = src(cv::Rect(0, i * 800, 640, 800));
		std::string str = images_path + "result_" + std::to_string(i + 1) + ".png";
		cv::imwrite(str, dst);
	}

	return 0;
}
