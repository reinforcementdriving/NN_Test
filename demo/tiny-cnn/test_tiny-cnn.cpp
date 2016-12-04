#include "funset.hpp"
#include <iostream>

int main()
{
	auto ret = test_dnn_mnist_predict();
	if (ret == 0) std::cout << "test ok" << std::endl;
	else std::cout << "test fail" << std::endl;

	return 0;
}

