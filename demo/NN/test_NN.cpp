#include <iostream>
#include "funset.hpp"

int main()
{
	int ret = test_CNN_predict();
	
	if (ret == 0) std::cout << "test ok!" << std::endl;
	else std::cout << "test fail" << std::endl;

	return 0;
}

