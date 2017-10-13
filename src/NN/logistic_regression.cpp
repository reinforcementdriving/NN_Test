#include "logistic_regression.hpp"
#include <fstream>
#include <algorithm>
#include <functional>
#include "common.hpp"

namespace ANN {

template<typename T>
int LogisticRegression<T>::init(const T* x, const T* y, int length, T learning_rate, T epsilon, int iterations)
{
	if (length < 2) {
		fprintf(stderr, "logistic regression train data is too little: %d\n", length);
		return -1;
	}

	this->learning_rate = learning_rate;
	this->epsilon = epsilon;
	this->iterations = iterations;

	this->length = length;
	this->x.resize(length);
	this->y.reset(new T[length]);

	for (int i = 0; i < length; ++i) {
		this->x[i].resize(2, (T)1);
		this->x[i][1] = x[i];
		this->y[i] = y[i];
	}

	return 0;
}

template<typename T>
int LogisticRegression<T>::train(const std::string& model)
{
	std::vector<T> w(2); // bias, weight
	generator_real_random_number<T>(w.data(), 1, (T)0, (T)0.000001);
	generator_real_random_number<T>(w.data()+1, 1, (T)0, (T)0.00001);

	int iter{ 0 };
	// http://blog.csdn.net/qq_27717921/article/details/54773061
	while (true) {
		for (int k = 0; k < w.size(); ++k) {
			T gradient{ 0 };

			for (int i = 0; i < x.size(); ++i) {
				gradient += (sigmoid(w[k] * x[i][k]) - y[i]) * x[i][k];
			}

			w[k] = w[k] + this->learning_rate * (gradient / x.size());
		}

		++iter;
		if (iter >= this->iterations) break;
	}

	this->weight = w[1];
	this->bias = w[0];
	fprintf(stdout, "weight: %f, bias: %f, iter: %d\n", this->weight, this->bias, iter);

	CHECK(store_model(model) == 0);

	return 0;
}

template<typename T>
int LogisticRegression<T>::load_model(const std::string& model) const
{
	std::ifstream file;
	file.open(model.c_str(), std::ios::binary);
	if (!file.is_open()) {
		fprintf(stderr, "open file fail: %s\n", model.c_str());
		return -1;
	}

	file.read((char*)&weight, sizeof(weight)* 1);
	file.read((char*)&bias, sizeof(bias)* 1);

	file.close();

	return 0;
}

template<typename T>
T LogisticRegression<T>::predict(T x) const
{
	return (sigmoid(weight * x + bias));
}

template<typename T>
int LogisticRegression<T>::store_model(const std::string& model) const
{
	std::ofstream file;
	file.open(model.c_str(), std::ios::binary);
	if (!file.is_open()) {
		fprintf(stderr, "open file fail: %s\n", model.c_str());
		return -1;
	}

	file.write((char*)&weight, sizeof(weight));
	file.write((char*)&bias, sizeof(bias));

	file.close();

	return 0;
}

template<typename T>
T LogisticRegression<T>::sigmoid(T x) const
{
	return ((T)1 / ((T)1 + exp(-x)));
}

template<typename T>
T LogisticRegression<T>::norm(const std::vector<T>& v1, const std::vector<T>& v2) const
{
	CHECK(v1.size() == v2.size());

	T sum{ 0 };

	for (int i = 0; i < v1.size(); ++i) {
		T minus = v1[i] - v2[i];
		sum += (minus * minus);
	}

	return std::sqrt(sum);
}

template class LogisticRegression<float>;
template class LogisticRegression<double>;

} // namespace ANN

