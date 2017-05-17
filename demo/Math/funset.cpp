#include "funset.hpp"
#include <math.h>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

static double determinant_opencv(const std::vector<float>& vec)
{
	int length = std::sqrt(vec.size());
	cv::Mat mat(length, length, CV_32FC1, const_cast<float*>(vec.data()));

	// In OpenCV, for small matrices(rows=cols<=3),the direct method is used.
	// For larger matrices the function uses LU factorization with partial pivoting.
	return cv::determinant(mat);
}

template<typename _Tp>
static _Tp det(const std::vector<std::vector<_Tp>>& mat, int N)
{
	if (mat.size() != N) {
		fprintf(stderr, "mat must be square matrix\n");
		return -1;
	}
	for (int i = 0; i < mat.size(); ++i) {
		if (mat[i].size() != N) {
			fprintf(stderr, "mat must be square matrix\n");
			return -1;
		}
	}

	_Tp ret{ 0 };

	if (N == 1) return mat[0][0];

	if (N == 2) {
		return (mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]);
	} else {
		// first col
		for (int i = 0; i < N; ++i) {
			std::vector<std::vector<_Tp>> m(N - 1);
			std::vector<int> m_rows;
			for (int t = 0; t < N; ++t) {
				if (i != t) m_rows.push_back(t);
			}
			for (int x = 0; x < N - 1; ++x) {
				m[x].resize(N - 1);
				for (int y = 0; y < N - 1; ++y) {
					m[x][y] = mat[m_rows[x]][y + 1];
				}
			}
			int sign = (int)pow(-1, 1 + i + 1);
			ret += mat[i][0] * sign * det<_Tp>(m, N-1);
		}
	}

	return ret;
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
	double det2 = det<float>(arr, N);

	fprintf(stderr, "det1: %f, det2: %f\n", det1, det2);

	return 0;
}

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
