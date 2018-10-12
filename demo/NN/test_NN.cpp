#include <iostream>
#include "funset.hpp"
#include "opencv.hpp"
#include "libsvm.hpp"

int main()
{
	int ret = test_decision_tree_train();
	
	if (ret == 0) fprintf(stdout, "========== test success ==========\n");
	else fprintf(stderr, "########## test fail ##########\n");

	return 0;
}

