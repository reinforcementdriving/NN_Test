#ifndef NN_COMMON_HPP_
#define NN_COMMON_HPP_

#define PI 3.14159265358979323846

#define CHECK(x) { \
	if (x) {} \
	else { fprintf(stderr, "Check Failed: %s, file: %s, line: %d\n", #x, __FILE__, __LINE__); return -1; } \
}

int compare_file();

#endif // NN_COMMON_HPP_
