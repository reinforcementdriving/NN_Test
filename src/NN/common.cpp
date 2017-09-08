#include <fstream>
#include <iostream>
#include <string>
#include "common.hpp"

int compare_file(const std::string& name1, const std::string& name2)
{
	std::ifstream infile1;
	infile1.open(name1.c_str(), std::ios::in | std::ios::binary);
	if (!infile1.is_open()) {
		fprintf(stderr, "failed to open file\n");
		return -1;
	}

	std::ifstream infile2;
	infile2.open(name2.c_str(), std::ios::in | std::ios::binary);
	if (!infile2.is_open()) {
		fprintf(stderr, "failed to open file\n");
		return -1;
	}

	size_t length1 = 0, length2 = 0;

	infile1.read((char*)&length1, sizeof(size_t));
	infile2.read((char*)&length2, sizeof(size_t));

	if (length1 != length2) {
		fprintf(stderr, "their length is mismatch: required length: %d, actual length: %d\n", length1, length2);
		return -1;
	}

	double* data1 = new double[length1];
	double* data2 = new double[length2];

	for (int i = 0; i < length1; i++) {
		infile1.read((char*)&data1[i], sizeof(double));
		infile2.read((char*)&data2[i], sizeof(double));

		if (data1[i] != data2[i]) {
			fprintf(stderr, "no equal: %d: %f, %f\n", i, data1[i], data2[i]);
		}
	}

	delete[] data1;
	delete[] data2;

	infile1.close();
	infile2.close();
}

