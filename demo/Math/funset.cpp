#include "funset.hpp"
#include <math.h>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "common.hpp"

int test_norm()
{
	fprintf(stderr, "test norm with C++:\n");
	std::vector<int> norm_types{ 0, 1, 2 }; // 正无穷、L1、L2
	std::vector<std::string> str{ "Inf", "L1", "L2" };

	// 1. vector
	std::vector<float> vec1{ -2, 3, 1 };
	std::vector<std::vector<float>> tmp1(1);
	tmp1[0].resize(vec1.size());
	for (int i = 0; i < vec1.size(); ++i) {
		tmp1[0][i] = vec1[i];
	}

	for (int i = 0; i < str.size(); ++i) {
		double value{ 0.f };
		norm(tmp1, norm_types[i], &value);

		fprintf(stderr, "vector: %s: %f\n", str[i].c_str(), value);
	}

	// 2. matrix
	std::vector<float> vec2{ -3, 2, 0, 5, 6, 2, 7, 4, 8 };
	const int row_col{ 3 };
	std::vector<std::vector<float>> tmp2(row_col);
	for (int y = 0; y < row_col; ++y) {
		tmp2[y].resize(row_col);
		for (int x = 0; x < row_col; ++x) {
			tmp2[y][x] = vec2[y * row_col + x];
		}
	}

	for (int i = 0; i < str.size(); ++i) {
		double value{ 0.f };
		norm(tmp2, norm_types[i], &value);

		fprintf(stderr, "matrix: %s: %f\n", str[i].c_str(), value);
	}

	fprintf(stderr, "\ntest norm with opencv:\n");
	norm_types[0] = 1; norm_types[1] = 2; norm_types[2] = 4; // 正无穷、L1、L2
	cv::Mat mat1(1, vec1.size(), CV_32FC1, vec1.data());

	for (int i = 0; i < norm_types.size(); ++i) {
		double value = cv::norm(mat1, norm_types[i]);
		fprintf(stderr, "vector: %s: %f\n", str[i].c_str(), value);
	}

	cv::Mat mat2(row_col, row_col, CV_32FC1, vec2.data());
	for (int i = 0; i < norm_types.size(); ++i) {
		double value = cv::norm(mat2, norm_types[i]);
		fprintf(stderr, "matrix: %s: %f\n", str[i].c_str(), value);
	}

	return 0;
}

int test_inverse_matrix()
{
	std::vector<float> vec{ 5, -2, 2, 7, 1, 0, 0, 3, -3, 1, 5, 0, 3, -1, -9, 4 };
	const int N{ 4 };
	if (vec.size() != (int)pow(N, 2)) {
		fprintf(stderr, "vec must be N^2\n");
		return -1;
	}

	std::vector<std::vector<float>> arr(N);
	for (int i = 0; i < N; ++i) {
		arr[i].resize(N);

		for (int j = 0; j < N; ++j) {
			arr[i][j] = vec[i * N + j];
		}
	}

	std::vector<std::vector<float>> inv1;
	int ret = inverse<float>(arr, inv1, N);

	fprintf(stderr, "source matrix: \n");
	print_matrix<float>(arr);
	fprintf(stderr, "c++ inverse matrix: \n");
	print_matrix<float>(inv1);

	cv::Mat mat(N, N, CV_32FC1, vec.data());
	cv::Mat inv2 = mat.inv();
	fprintf(stderr, "opencv inverse matrix: \n");
	print_matrix(inv2);

	return 0;
}

int test_adjoint_matrix()
{
	std::vector<float> vec{5, -2, 2, 7, 1, 0, 0, 3, -3, 1, 5, 0, 3, -1, -9, 4 };
	const int N{ 4 };
	if (vec.size() != (int)pow(N, 2)) {
		fprintf(stderr, "vec must be N^2\n");
		return -1;
	}

	std::vector<std::vector<float>> arr(N);
	for (int i = 0; i < N; ++i) {
		arr[i].resize(N);

		for (int j = 0; j < N; ++j) {
			arr[i][j] = vec[i * N + j];
		}
	}

	std::vector<std::vector<float>> adj;
	int ret = adjoint<float>(arr, adj, N);

	fprintf(stderr, "source matrix: \n");
	print_matrix<float>(arr);
	fprintf(stderr, "adjoint matrx: \n");
	print_matrix<float>(adj);

	return 0;
}

static double determinant_opencv(const std::vector<float>& vec)
{
	int length = std::sqrt(vec.size());
	cv::Mat mat(length, length, CV_32FC1, const_cast<float*>(vec.data()));

	// In OpenCV, for small matrices(rows=cols<=3),the direct method is used.
	// For larger matrices the function uses LU factorization with partial pivoting.
	return cv::determinant(mat);
}

int test_determinant()
{
	std::vector<float> vec{ 1, 0, 2, -1, 3, 0, 0, 5, 2, 1, 4, -3, 1, 0, 5, 0};
	const int N{ 4 };
	if (vec.size() != (int)pow(N, 2)) {
		fprintf(stderr, "vec must be N^2\n");
		return -1;
	}
	double det1 = determinant_opencv(vec);

	std::vector<std::vector<float>> arr(N);
	for (int i = 0; i < N; ++i) {
		arr[i].resize(N);

		for (int j = 0; j < N; ++j) {
			arr[i][j] = vec[i * N + j];
		}
	}
	double det2 = determinant<float>(arr, N);

	fprintf(stderr, "det1: %f, det2: %f\n", det1, det2);

	return 0;
}

int test_matrix_transpose()
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
