#ifndef FBC_NN_LOGISTICREGRESSION_HPP_
#define FBC_NN_LOGISTICREGRESSION_HPP_

#include <string>
#include <memory>
#include <vector>

namespace ANN {

template<typename T>
class LogisticRegression { // two categories
public:
	LogisticRegression() = default;
	int init(const T* data, const T* labels, int train_num, int feature_length, T learning_rate = 0.00001, int iterations = 10000);
	int train(const std::string& model);
	int load_model(const std::string& model);
	T predict(const T* data, int feature_length) const; // y = 1/(1+exp(-(wx+b)))

private:
	int store_model(const std::string& model) const;
	T calc_sigmoid(T x) const; // y = 1/(1+exp(-x))
	T norm(const std::vector<T>& v1, const std::vector<T>& v2) const;
	void batch_gradient_descent();
	void mini_batch_gradient_descent();

	std::vector<std::vector<T>> data;
	std::vector<T> labels;
	int iterations = 10000;
	int train_num = 0;
	int feature_length = 0;
	T learning_rate = 0.00001;
	std::vector<T> thetas; // coefficient
	//T epsilon = 0.0001; // termination condition
};

} // namespace ANN

#endif // FBC_NN_LOGISTICREGRESSION_HPP_
