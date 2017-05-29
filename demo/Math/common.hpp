#ifndef FBC_MATH_COMMON_HPP_
#define FBC_MATH_COMMON_HPP_

#include <math.h>
#include <vector>
#include <limits>
#include <opencv2/opencv.hpp>

#define EXP 1.0e-5

// 求特征值和特征向量
template<typename _Tp>
static inline _Tp hypot(_Tp a, _Tp b)
{
	a = std::abs(a);
	b = std::abs(b);
	if (a > b) {
		b /= a;
		return a*std::sqrt(1 + b*b);
	}
	if (b > 0) {
		a /= b;
		return b*std::sqrt(1 + a*a);
	}
	return 0;
}

template<typename _Tp>
int eigen(const std::vector<std::vector<_Tp>>& mat, std::vector<_Tp>& eigenvalues, std::vector<std::vector<_Tp>>& eigenvectors, bool sort_ = true)
{
	auto n = mat.size();
	for (const auto& m : mat) {
		if (m.size() != n) {
			fprintf(stderr, "mat must be square and it should be a real symmetric matrix\n");
			return -1;
		}
	}

	eigenvalues.resize(n, (_Tp)0);
	std::vector<_Tp> V(n*n, (_Tp)0);
	for (int i = 0; i < n; ++i) {
		V[n * i + i] = (_Tp)1;
		eigenvalues[i] = mat[i][i];
	}

	const _Tp eps = std::numeric_limits<_Tp>::epsilon();
	int maxIters{ (int)n * (int)n * 30 };
	_Tp mv{ (_Tp)0 };
	std::vector<int> indR(n, 0), indC(n, 0);
	std::vector<_Tp> A;
	for (int i = 0; i < n; ++i) {
		A.insert(A.begin() + i * n, mat[i].begin(), mat[i].end());
	}

	for (int k = 0; k < n; ++k) {
		int m, i;
		if (k < n - 1) {
			for (m = k + 1, mv = std::abs(A[n*k + m]), i = k + 2; i < n; i++) {
				_Tp val = std::abs(A[n*k + i]);
				if (mv < val)
					mv = val, m = i;
			}
			indR[k] = m;
		}
		if (k > 0) {
			for (m = 0, mv = std::abs(A[k]), i = 1; i < k; i++) {
				_Tp val = std::abs(A[n*i + k]);
				if (mv < val)
					mv = val, m = i;
			}
			indC[k] = m;
		}
	}

	if (n > 1) for (int iters = 0; iters < maxIters; iters++) {
		int k, i, m;
		// find index (k,l) of pivot p
		for (k = 0, mv = std::abs(A[indR[0]]), i = 1; i < n - 1; i++) {
			_Tp val = std::abs(A[n*i + indR[i]]);
			if (mv < val)
				mv = val, k = i;
		}
		int l = indR[k];
		for (i = 1; i < n; i++) {
			_Tp val = std::abs(A[n*indC[i] + i]);
			if (mv < val)
				mv = val, k = indC[i], l = i;
		}

		_Tp p = A[n*k + l];
		if (std::abs(p) <= eps)
			break;
		_Tp y = (_Tp)((eigenvalues[l] - eigenvalues[k])*0.5);
		_Tp t = std::abs(y) + hypot(p, y);
		_Tp s = hypot(p, t);
		_Tp c = t / s;
		s = p / s; t = (p / t)*p;
		if (y < 0)
			s = -s, t = -t;
		A[n*k + l] = 0;

		eigenvalues[k] -= t;
		eigenvalues[l] += t;

		_Tp a0, b0;

#undef rotate
#define rotate(v0, v1) a0 = v0, b0 = v1, v0 = a0*c - b0*s, v1 = a0*s + b0*c

		// rotate rows and columns k and l
		for (i = 0; i < k; i++)
			rotate(A[n*i + k], A[n*i + l]);
		for (i = k + 1; i < l; i++)
			rotate(A[n*k + i], A[n*i + l]);
		for (i = l + 1; i < n; i++)
			rotate(A[n*k + i], A[n*l + i]);

		// rotate eigenvectors
		for (i = 0; i < n; i++)
			rotate(V[n*k+i], V[n*l+i]);

#undef rotate

		for (int j = 0; j < 2; j++) {
			int idx = j == 0 ? k : l;
			if (idx < n - 1) {
				for (m = idx + 1, mv = std::abs(A[n*idx + m]), i = idx + 2; i < n; i++) {
					_Tp val = std::abs(A[n*idx + i]);
					if (mv < val)
						mv = val, m = i;
				}
				indR[idx] = m;
			}
			if (idx > 0) {
				for (m = 0, mv = std::abs(A[idx]), i = 1; i < idx; i++) {
					_Tp val = std::abs(A[n*i + idx]);
					if (mv < val)
						mv = val, m = i;
				}
				indC[idx] = m;
			}
		}
	}

	// sort eigenvalues & eigenvectors
	if (sort_) {
		for (int k = 0; k < n - 1; k++) {
			int m = k;
			for (int i = k + 1; i < n; i++) {
				if (eigenvalues[m] < eigenvalues[i])
					m = i;
			}
			if (k != m) {
				std::swap(eigenvalues[m], eigenvalues[k]);
				for (int i = 0; i < n; i++)
					std::swap(V[n*m+i], V[n*k+i]);
			}
		}
	}

	eigenvectors.resize(n);
	for (int i = 0; i < n; ++i) {
		eigenvectors[i].resize(n);
		eigenvectors[i].assign(V.begin() + i * n, V.begin() + i * n + n);
	}

	return 0;
}

// 求范数
typedef enum Norm_Types_ {
	Norm_INT = 0, // 无穷大
	Norm_L1, // L1
	Norm_L2 // L2
} Norm_Types;

template<typename _Tp>
int norm(const std::vector<std::vector<_Tp>>& mat, int type, double* value)
{
	*value = 0.f;

	switch (type) {
		case Norm_INT: {
			for (int i = 0; i < mat.size(); ++i) {
				for (const auto& t : mat[i]) {
					*value = std::max(*value, (double)(fabs(t)));
				}
			}
		}
			break;
		case Norm_L1: {
			for (int i = 0; i < mat.size(); ++i) {
				for (const auto& t : mat[i]) {
					*value += (double)(fabs(t));
				}
			}
		}
			break;
		case Norm_L2: {
			for (int i = 0; i < mat.size(); ++i) {
				for (const auto& t : mat[i]) {
					*value += t * t;
				}
			}
			*value = std::sqrt(*value);
		}
			break;
		default: {
			fprintf(stderr, "norm type is not supported\n");
			return -1;
		}
	}

	return 0;
}

// 计算行列式
template<typename _Tp>
_Tp determinant(const std::vector<std::vector<_Tp>>& mat, int N)
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
	}
	else {
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
			ret += mat[i][0] * sign * determinant<_Tp>(m, N - 1);
		}
	}

	return ret;
}

// 计算伴随矩阵
template<typename _Tp>
int adjoint(const std::vector<std::vector<_Tp>>& mat, std::vector<std::vector<_Tp>>& adj, int N)
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

	adj.resize(N);
	for (int i = 0; i < N; ++i) {
		adj[i].resize(N);
	}

	for (int y = 0; y < N; ++y) {
		std::vector<int> m_cols;
		for (int i = 0; i < N; ++i) {
			if (i != y) m_cols.push_back(i);
		}

		for (int x = 0; x < N; ++x) {
			std::vector<int> m_rows;
			for (int i = 0; i < N; ++i) {
				if (i != x) m_rows.push_back(i);
			}

			std::vector<std::vector<_Tp>> m(N - 1);
			for (int i = 0; i < N - 1; ++i) {
				m[i].resize(N - 1);
			}
			for (int j = 0; j < N - 1; ++j) {
				for (int i = 0; i < N - 1; ++i) {
					m[j][i] = mat[m_rows[j]][m_cols[i]];
				}
			}

			int sign = (int)pow(-1, x + y);
			adj[y][x] = sign * determinant<_Tp>(m, N-1);
		}
	}

	return 0;
}

template<typename _Tp>
void print_matrix(const std::vector<std::vector<_Tp>>& mat)
{
	int rows = mat.size();
	for (int y = 0; y < rows; ++y) {
		for (int x = 0; x < mat[y].size(); ++x) {
			fprintf(stderr, "  %f  ", mat[y][x]);
		}
		fprintf(stderr, "\n");
	}
	fprintf(stderr, "\n");
}

void print_matrix(const cv::Mat& mat)
{
	assert(mat.channels() == 1);

	for (int y = 0; y < mat.rows; ++y) {
		for (int x = 0; x < mat.cols; ++x) {
			if (mat.depth() == CV_8U) {
				unsigned char value = mat.at<uchar>(y, x);
				fprintf(stderr, "  %d  ", value);
			}
			else if (mat.depth() == CV_32F) {
				float value = mat.at<float>(y, x);
				fprintf(stderr, "  %f  ", value);
			}
			else if (mat.depth() == CV_64F) {
				double value = mat.at<double>(y, x);
				fprintf(stderr, "  %f  ", value);
			}
			else {
				fprintf(stderr, "don't support type: %d\n", mat.depth());
				return;
			}
		}
		fprintf(stderr, "\n");
	}
	fprintf(stderr, "\n");
}

// 求逆矩阵
template<typename _Tp>
int inverse(const std::vector<std::vector<_Tp>>& mat, std::vector<std::vector<_Tp>>& inv, int N)
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

	_Tp det = determinant(mat, N);
	if (fabs(det) < EXP) {
		fprintf(stderr, "mat's determinant don't equal 0\n");
		return -1;
	}

	inv.resize(N);
	for (int i = 0; i < N; ++i) {
		inv[i].resize(N);
	}

	double coef = 1.f / det;
	std::vector<std::vector<_Tp>> adj;
	if (adjoint(mat, adj, N) != 0) return -1;

	for (int y = 0; y < N; ++y) {
		for (int x = 0; x < N; ++x) {
			inv[y][x] = (_Tp)(coef * adj[y][x]);
		}
	}

	return 0;
}

#endif // FBC_MATH_COMMON_HPP_
