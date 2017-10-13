#ifndef FBC_NN_LOGISTICREGRESSION_HPP_
#define FBC_NN_LOGISTICREGRESSION_HPP_

#include <string>
#include <memory>
#include <vector>

namespace ANN {

template<typename T>
class LogisticRegression {
public:
	LogisticRegression() = default;
	int init(const T* x, const T* y, int length, T learning_rate = 0.00001, T epsilon = 0.0001, int iterations = 10000);
	int train(const std::string& model);
	int load_model(const std::string& model) const;
	T predict(T x) const; // y = 1/(1+exp(-(wx+b)))

private:
	int store_model(const std::string& model) const;
	T sigmoid(T x) const; // y = 1/(1+exp(-x))
	T norm(const std::vector<T>& v1, const std::vector<T>& v2) const;

	std::vector<std::vector<T>> x;
	std::unique_ptr<T[]> y;
	int iterations = 10000;
	int length = 0;
	T learning_rate = 0.00001;
	T epsilon = 0.0001; // termination condition
	T weight = 0;
	T bias = 0;
};

} // namespace ANN

#endif // FBC_NN_LOGISTICREGRESSION_HPP_
