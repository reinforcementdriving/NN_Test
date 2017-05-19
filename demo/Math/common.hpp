#ifndef FBC_MATH_COMMON_HPP_
#define FBC_MATH_COMMON_HPP_

#include <math.h>
#include <vector>

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

#endif // FBC_MATH_COMMON_HPP_
