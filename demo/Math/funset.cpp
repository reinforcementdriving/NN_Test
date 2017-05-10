#include "funset.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

int test_mat_transpose()
{
	const std::vector<std::string> image_name{ "E:/GitCode/NN_Test/data/images/test1.jpg",
		"E:/GitCode/NN_Test/data/images/ret_mat_transpose.jpg"};
	cv::Mat mat_src = cv::imread(image_name[0]);
	if (!mat_src.data) {
		fprintf(stderr, "read image fail: %s\n", image_name[0].c_str());
		return -1;
	}

	cv::Mat mat_dst(mat_src.cols, mat_src.rows, mat_src.type());

	for (int h = 0; h < mat_dst.rows; ++h) {
		for (int w = 0; w < mat_dst.cols; ++w) {
			const cv::Vec3b& s = mat_src.at<cv::Vec3b>(w, h);
			cv::Vec3b& d = mat_dst.at<cv::Vec3b>(h, w);
			d = s;
		}
	}

	cv::imwrite(image_name[1], mat_dst);

	return 0;
}
