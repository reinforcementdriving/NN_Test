#include <iostream>
#include "funset.hpp"

int main()
{
	int ret = test_dnn_mnist_predict();
	if (ret == 0) fprintf(stderr, "test success\n");
	else fprintf(stderr, "test fail\n");

	return 0;
}
