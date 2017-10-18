#include "logistic_regression.hpp"
#include <fstream>
#include <algorithm>
#include <functional>
#include <numeric>
#include "common.hpp"

namespace ANN {

template<typename T>
int LogisticRegression<T>::init(const T* data, const T* labels, int train_num, int feature_length, T learning_rate = 0.00001, int iterations = 10000)
{
	if (train_num < 2) {
		fprintf(stderr, "logistic regression train samples num is too little: %d\n", train_num);
		return -1;
	}
	if (learning_rate <= 0) {
		fprintf(stderr, "learning rate must be greater 0: %f\n", learning_rate);
		return -1;
	}
	if (iterations <= 0) {
		fprintf(stderr, "number of iterations cannot be zero or a negative number: %d\n", iterations);
		return -1;
	}

	this->learning_rate = learning_rate;
	this->iterations = iterations;

	this->train_num = train_num;
	this->feature_length = feature_length;

	this->data.resize(train_num);
	this->labels.resize(train_num);

	for (int i = 0; i < train_num; ++i) {
		const T* p = data + i * feature_length;
		this->data[i].resize(feature_length+1);
		this->data[i][0] = (T)1; // bias

		for (int j = 0; j < feature_length; ++j) {
			this->data[i][j+1] = p[j];
		}

		this->labels[i] = labels[i];
	}

	this->thetas.resize(feature_length + 1, (T)0.); // bias + feature_length

	return 0;
}

template<typename T>
int LogisticRegression<T>::train(const std::string& model)
{
	CHECK(data.size() == labels.size());

	// gradient descent
	for (int i = 0; i < iterations; ++i) {
		std::unique_ptr<T[]> z(new T[train_num]), gradient(new T[thetas.size()]);
		for (int j = 0; j < train_num; ++j) {
			z.get()[j] = (T)0.;
			for (int t = 0; t < feature_length+1; ++t) {
				z.get()[j] += data[j][t] * thetas[t];
			}
		}

		std::unique_ptr<T[]> pcal_a(new T[train_num]), pcal_b(new T[train_num]), pcal_ab(new T[train_num]);
		for (int j = 0; j < train_num; ++j) {
			pcal_a.get()[j] = calc_sigmoid(z.get()[j]) - labels[j];
			pcal_b.get()[j] = data[j][0]; // bias
			pcal_ab.get()[j] = pcal_a.get()[j] * pcal_b.get()[j];
		}

		gradient.get()[0] = ((T)1. / train_num) * std::accumulate(pcal_ab.get(), pcal_ab.get()+train_num, (T)0.); // bias

		for (int j = 1; j < thetas.size(); ++j) {
			for (int t = 0; t < train_num; ++t) {
				pcal_b.get()[t] = data[t][j];
				pcal_ab.get()[t] = pcal_a.get()[t] * pcal_b.get()[t];
			}

			gradient.get()[j] = ((T)1. / train_num) * std::accumulate(pcal_ab.get(), pcal_ab.get() + train_num, (T)0.);
		}

		for (int i = 0; i < thetas.size(); ++i) {
			thetas[i] = thetas[i] - learning_rate / train_num * gradient.get()[i];
		}
	}

	CHECK(store_model(model) == 0);

	return 0;
}

template<typename T>
int LogisticRegression<T>::load_model(const std::string& model)
{
	std::ifstream file;
	file.open(model.c_str(), std::ios::binary);
	if (!file.is_open()) {
		fprintf(stderr, "open file fail: %s\n", model.c_str());
		return -1;
	}

	int length{ 0 };
	file.read((char*)&length, sizeof(length));
	thetas.resize(length);
	file.read((char*)thetas.data(), sizeof(T)*thetas.size());

	file.close();

	return 0;
}

template<typename T>
T LogisticRegression<T>::predict(const T* data, int feature_length) const
{
	CHECK(feature_length + 1 == thetas.size());

	T value{(T)0.};
	for (int t = 1; t < thetas.size(); ++t) {
		value += data[t - 1] * thetas[t];
	}
	return (calc_sigmoid(value + thetas[0]));
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

	int length = thetas.size();
	file.write((char*)&length, sizeof(length));
	file.write((char*)thetas.data(), sizeof(T) * thetas.size());

	file.close();

	return 0;
}

template<typename T>
T LogisticRegression<T>::calc_sigmoid(T x) const
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

