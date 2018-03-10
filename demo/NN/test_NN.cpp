#include <iostream>
#include "funset.hpp"
#include "opencv.hpp"
#include "libsvm.hpp"
#include "common.hpp"

int main()
{
	int ret = test_kmeans();
	
	if (ret == 0) std::cout << "========== test success ==========" << std::endl;
	else std::cerr << "########## test fail ##########" << std::endl;

	return 0;
}

