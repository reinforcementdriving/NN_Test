#ifndef FBC_NN_COMMON_HPP_
#define FBC_NN_COMMON_HPP_

#define PI 3.14159265358979323846

#define CHECK(x) { \
	if (x) {} \
	else { fprintf(stderr, "Check Failed: %s, file: %s, line: %d\n", #x, __FILE__, __LINE__); return -1; } \
}

template<typename T>
void generator_real_random_number(T* data, int length, T a = (T)0, T b = (T)1);

int compare_file();
int mat_horizontal_concatenate();

#endif // FBC_NN_COMMON_HPP_
