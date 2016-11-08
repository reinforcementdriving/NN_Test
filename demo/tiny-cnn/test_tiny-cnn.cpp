#include "funset.hpp"
#include <iostream>

int main()
{
	auto ret = test_tiny_cnn_train();
	if (ret == 0) std::cout << "test ok" << std::endl;
	else std::cout << "test fail" << std::endl;

	return 0;
}

