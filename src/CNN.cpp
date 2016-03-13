#include <assert.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <numeric>
#include <windows.h>
#include <random>
#include <algorithm>

#include <CNN.hpp>

namespace ANN {

CNN::CNN()
{
	data_input_train = NULL;
	data_output_train = NULL;
	data_input_test = NULL;
	data_output_test = NULL;
	data_single_image = NULL;
	data_single_label = NULL;
}

CNN::~CNN()
{
	release();
}

void CNN::release()
{
	if (data_input_train) {
		delete[] data_input_train;
		data_input_train = NULL;
	}

	if (data_output_train) {
		delete[] data_output_train;
		data_output_train = NULL;
	}

	if (data_input_test) {
		delete[] data_input_test;
		data_input_test = NULL;
	}

	if (data_output_test) {
		delete[] data_output_test;
		data_output_test = NULL;
	}
}

void CNN::init_variable(float* val, float c, int len)
{
	for (int i = 0; i < len; i++) {
		val[i] = c;
	}
}

void CNN::init()
{
	int len1 = width_image_input_CNN * height_image_input_CNN * num_patterns_train_CNN;
	data_input_train = new float[len1];
	init_variable(data_input_train, -1.0, len1);

	int len2 = num_map_output_CNN * num_patterns_train_CNN;
	data_output_train = new float[len2];
	init_variable(data_output_train, -0.9, len2);

	int len3 = width_image_input_CNN * height_image_input_CNN * num_patterns_test_CNN;
	data_input_test = new float[len3];
	init_variable(data_input_test, -1.0, len3);

	int len4 = num_map_output_CNN * num_patterns_test_CNN;
	data_output_test = new float[len4];
	init_variable(data_output_test, -0.9, len4);

	initWeightThreshold();
	getSrcData();
}

float CNN::uniform_rand(float min, float max)
{
	static std::mt19937 gen(1);
	std::uniform_real_distribution<float> dst(min, max);
	return dst(gen);
}

bool CNN::uniform_rand(float* src, int len, float min, float max)
{
	for (int i = 0; i < len; i++) {
		src[i] = uniform_rand(min, max);
	}

	return true;
}

bool CNN::initWeightThreshold()
{
	srand(time(0) + rand());
	const float scale = 6.0;

	//const float_t weight_base = std::sqrt(scale_ / (fan_in + fan_out));
	//fan_in = width_kernel_conv_CNN * height_kernel_conv_CNN * num_map_input_CNN = 5 * 5 * 1
	//fan_out = width_kernel_conv_CNN * height_kernel_conv_CNN * num_map_C1_CNN = 5 * 5 * 6
	float min_ = -std::sqrt(scale / (25.0 + 150.0));
	float max_ = std::sqrt(scale / (25.0 + 150.0));
	uniform_rand(weight_C1, len_weight_C1_CNN, min_, max_);
	//for (int i = 0; i < len_weight_C1_CNN; i++) {
	//	weight_C1[i] = -1 + 2 * ((float)rand()) / RAND_MAX; //[-1, 1]
	//}
	for (int i = 0; i < len_bias_C1_CNN; i++) {
		bias_C1[i] = -1 + 2 * ((float)rand()) / RAND_MAX;//0.0;//
	}

	min_ = -std::sqrt(scale / (4.0 + 1.0));
	max_ = std::sqrt(scale / (4.0 + 1.0));
	uniform_rand(weight_S2, len_weight_S2_CNN, min_, max_);
	//for (int i = 0; i < len_weight_S2_CNN; i++) {
	//	weight_S2[i] = -1 + 2 * ((float)rand()) / RAND_MAX;
	//}
	for (int i = 0; i < len_bias_S2_CNN; i++) {
		bias_S2[i] = -1 + 2 * ((float)rand()) / RAND_MAX;//0.0;// 
	}

	min_ = -std::sqrt(scale / (150.0 + 400.0));
	max_ = std::sqrt(scale / (150.0 + 400.0));
	uniform_rand(weight_C3, len_weight_C3_CNN, min_, max_);
	//for (int i = 0; i < len_weight_C3_CNN; i++) {
	//	weight_C3[i] = -1 + 2 * ((float)rand()) / RAND_MAX;
	//}
	for (int i = 0; i < len_bias_C3_CNN; i++) {
		bias_C3[i] = -1 + 2 * ((float)rand()) / RAND_MAX;//0.0;// 
	}

	min_ = -std::sqrt(scale / (4.0 + 1.0));
	max_ = std::sqrt(scale / (4.0 + 1.0));
	uniform_rand(weight_S4, len_weight_S4_CNN, min_, max_);
	//for (int i = 0; i < len_weight_S4_CNN; i++) {
	//	weight_S4[i] = -1 + 2 * ((float)rand()) / RAND_MAX;
	//}
	for (int i = 0; i < len_bias_S4_CNN; i++) {
		bias_S4[i] = -1 + 2 * ((float)rand()) / RAND_MAX; //0.0;//
	}

	min_ = -std::sqrt(scale / (400.0 + 3000.0));
	max_ = std::sqrt(scale / (400.0 + 3000.0));
	uniform_rand(weight_C5, len_weight_C5_CNN, min_, max_);
	//for (int i = 0; i < len_weight_C5_CNN; i++) {
	//	weight_C5[i] = -1 + 2 * ((float)rand()) / RAND_MAX;
	//}
	for (int i = 0; i < len_bias_C5_CNN; i++) {
		bias_C5[i] =-1 + 2 * ((float)rand()) / RAND_MAX; //0.0;// 
	}

	min_ = -std::sqrt(scale / (120.0 + 10.0));
	max_ = std::sqrt(scale / (120.0 + 10.0));
	uniform_rand(weight_output, len_weight_output_CNN, min_, max_);
	//for (int i = 0; i < len_weight_output_CNN; i++) {
	//	weight_output[i] = -1 + 2 * ((float)rand()) / RAND_MAX;
	//}
	for (int i = 0; i < len_bias_output_CNN; i++) {
		bias_output[i] = -1 + 2 * ((float)rand()) / RAND_MAX;//0.0;// 
	}

	return true;
}

static int reverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

static void readMnistImages(std::string filename, float* data_dst, int num_image)
{
	const int width_src_image = 28;
	const int height_src_image = 28;
	const int x_padding = 2;
	const int y_padding = 2;
	const float scale_min = -1;
	const float scale_max = 1;

	std::ifstream file(filename, std::ios::binary);
	assert(file.is_open());

	int magic_number = 0;
	int number_of_images = 0;
	int n_rows = 0;
	int n_cols = 0;
	file.read((char*)&magic_number, sizeof(magic_number));
	magic_number = reverseInt(magic_number);
	file.read((char*)&number_of_images, sizeof(number_of_images));
	number_of_images = reverseInt(number_of_images);
	assert(number_of_images == num_image);
	file.read((char*)&n_rows, sizeof(n_rows));
	n_rows = reverseInt(n_rows);
	file.read((char*)&n_cols, sizeof(n_cols));
	n_cols = reverseInt(n_cols);
	assert(n_rows == height_src_image && n_cols == width_src_image);

	int size_single_image = width_image_input_CNN * height_image_input_CNN;

	for (int i = 0; i < number_of_images; ++i) {
		int addr = size_single_image * i;

		for (int r = 0; r < n_rows; ++r) {
			for (int c = 0; c < n_cols; ++c) {
				unsigned char temp = 0;
				file.read((char*)&temp, sizeof(temp));
				data_dst[addr + width_image_input_CNN * (r + y_padding) + c + x_padding] = (temp / 255.0) * (scale_max - scale_min) + scale_min;
			}
		}
	}
}

static void readMnistLabels(std::string filename, float* data_dst, int num_image)
{
	const float scale_min = -0.9;
	const float scale_max = 0.9;

	std::ifstream file(filename, std::ios::binary);
	assert(file.is_open());

	int magic_number = 0;
	int number_of_images = 0;
	file.read((char*)&magic_number, sizeof(magic_number));
	magic_number = reverseInt(magic_number);
	file.read((char*)&number_of_images, sizeof(number_of_images));
	number_of_images = reverseInt(number_of_images);
	assert(number_of_images == num_image);

	for (int i = 0; i < number_of_images; ++i) {
		unsigned char temp = 0;
		file.read((char*)&temp, sizeof(temp));
		data_dst[i * num_map_output_CNN + temp] = scale_max;
	}
}

bool CNN::getSrcData()
{
	assert(data_input_train && data_output_train && data_input_test && data_output_test);

	std::string filename_train_images = "E:/GitCode/NN_Test/data/train-images.idx3-ubyte";
	std::string filename_train_labels = "E:/GitCode/NN_Test/data/train-labels.idx1-ubyte";
	readMnistImages(filename_train_images, data_input_train, num_patterns_train_CNN);
	/*unsigned char* p = new unsigned char[num_neuron_input_CNN];
	memset(p, 0, sizeof(unsigned char) * num_neuron_input_CNN);
	for (int j = 0, i = 59998 * num_neuron_input_CNN; j< num_neuron_input_CNN; j++, i++) {
		p[j] = (unsigned char)((data_input_train[i] + 1.0) / 2.0 * 255.0);
	}
	delete[] p;*/
	readMnistLabels(filename_train_labels, data_output_train, num_patterns_train_CNN);
	/*float* q = new float[num_neuron_output_CNN];
	memset(q, 0, sizeof(float) * num_neuron_output_CNN);
	for (int j = 0, i = 59998 * num_neuron_output_CNN; j < num_neuron_output_CNN; j++, i++) {
		q[j] = data_output_train[i];
	}
	delete[] q;*/

	std::string filename_test_images = "E:/GitCode/NN_Test/data/t10k-images.idx3-ubyte";
	std::string filename_test_labels = "E:/GitCode/NN_Test/data/t10k-labels.idx1-ubyte";
	readMnistImages(filename_test_images, data_input_test, num_patterns_test_CNN);
	readMnistLabels(filename_test_labels, data_output_test, num_patterns_test_CNN);

	return true;
}

bool CNN::train()
{
	out2wi_S2.clear();
	out2bias_S2.clear();
	out2wi_S4.clear();
	out2bias_S4.clear();
	in2wo_C3.clear();
	weight2io_C3.clear();
	bias2out_C3.clear();
	in2wo_C1.clear();
	weight2io_C1.clear();
	bias2out_C1.clear();

	calc_out2wi(width_image_C1_CNN, height_image_C1_CNN, width_image_S2_CNN, height_image_S2_CNN, num_map_S2_CNN, out2wi_S2);
	calc_out2bias(width_image_S2_CNN, height_image_S2_CNN, num_map_S2_CNN, out2bias_S2);
	calc_out2wi(width_image_C3_CNN, height_image_C3_CNN, width_image_S4_CNN, height_image_S4_CNN, num_map_S4_CNN, out2wi_S4);
	calc_out2bias(width_image_S4_CNN, height_image_S4_CNN, num_map_S4_CNN, out2bias_S4);
	calc_in2wo(width_image_C3_CNN, height_image_C3_CNN, width_image_S4_CNN, height_image_S4_CNN, num_map_C3_CNN, num_map_S4_CNN, in2wo_C3);
	calc_weight2io(width_image_C3_CNN, height_image_C3_CNN, width_image_S4_CNN, height_image_S4_CNN, num_map_C3_CNN, num_map_S4_CNN, weight2io_C3);
	calc_bias2out(width_image_C3_CNN, height_image_C3_CNN, width_image_S4_CNN, height_image_S4_CNN, num_map_C3_CNN, num_map_S4_CNN, bias2out_C3);
	calc_in2wo(width_image_C1_CNN, height_image_C1_CNN, width_image_S2_CNN, height_image_S2_CNN, num_map_C1_CNN, num_map_C3_CNN, in2wo_C1);
	calc_weight2io(width_image_C1_CNN, height_image_C1_CNN, width_image_S2_CNN, height_image_S2_CNN, num_map_C1_CNN, num_map_C3_CNN, weight2io_C1);
	calc_bias2out(width_image_C1_CNN, height_image_C1_CNN, width_image_S2_CNN, height_image_S2_CNN, num_map_C1_CNN, num_map_C3_CNN, bias2out_C1);

	int iter = 0;
	for (iter = 0; iter < num_epochs_CNN; iter++) {
		std::cout << "epoch: " << iter;

		float accuracyRate = test();//0;
		std::cout << ",    accuray rate: " << accuracyRate << std::endl;
		if (accuracyRate > accuracy_rate_CNN) {
			saveModelFile("E:/GitCode/NN_Test/data/cnn.model");
			std::cout << "generate cnn model" << std::endl;
			break;
		}

		for (int i = 0; i < num_patterns_train_CNN; i++) {
			data_single_image = data_input_train + i * num_neuron_input_CNN;
			data_single_label = data_output_train + i * num_neuron_output_CNN;

			Forward_C1();
			Forward_S2();
			Forward_C3();
			Forward_S4();
			Forward_C5();
			Forward_output();

			Backward_output();
			Backward_C5();
			Backward_S4();
			Backward_C3();
			Backward_S2();
			Backward_C1();
			Backward_input();

			UpdateWeights();
		}
	}

	if (iter == num_epochs_CNN) {
		saveModelFile("E:/GitCode/NN_Test/data/cnn.model");
		std::cout << "generate cnn model" << std::endl;
	}

	return true;
}

float CNN::activation_function_tanh(float x)
{
	float ep = std::exp(x);
	float em = std::exp(-x);

	return (ep - em) / (ep + em);
}

float CNN::activation_function_tanh_derivative(float x)
{
	return (1.0 - x * x);
}

float CNN::activation_function_identity(float x)
{
	return x;
}

float CNN::activation_function_identity_derivative(float x)
{
	return 1;
}

float CNN::loss_function_mse(float y, float t)
{
	return (y - t) * (y - t) / 2;
}

float CNN::loss_function_mse_derivative(float y, float t)
{
	return (y - t);
}

void CNN::loss_function_gradient(const float* y, const float* t, float* dst, int len)
{
	for (int i = 0; i < len; i++) {
		dst[i] = loss_function_mse_derivative(y[i], t[i]);
	}
}

float CNN::dot_product(const float* s1, const float* s2, int len)
{
	float result = 0.0;

	for (int i = 0; i < len; i++) {
		result += s1[i] * s2[i];
	}

	return result;
}

bool CNN::muladd(const float* src, float c, int len, float* dst)
{
	for (int i = 0; i < len; i++) {
		dst[i] += (src[i] * c);
	}

	return true;
}

int CNN::get_index(int x, int y, int channel, int width, int height, int depth)
{
	assert(x >= 0 && x < width);
	assert(y >= 0 && y < height);
	assert(channel >= 0 && channel < depth);
	return (height * channel + y) * width + x;
}

bool CNN::Forward_C1()
{
	init_variable(neuron_C1, 0.0, num_neuron_C1_CNN);

	/*for (int i = 0; i < num_map_C1_CNN; i++) {
		int addr1 = i * width_image_C1_CNN * height_image_C1_CNN;
		int addr2 = i * width_kernel_conv_CNN * height_kernel_conv_CNN;
		float* image = &neuron_C1[0] + addr1;
		const float* weight = &weight_C1[0] + addr2;

		for (int y = 0; y < height_image_C1_CNN; y++) {
			for (int x = 0; x < width_image_C1_CNN; x++) {
				float sum = 0.0;
				const float* image_input = data_single_image + y * width_image_input_CNN + x;

				for (int m = 0; m < height_kernel_conv_CNN; m++) {
					for (int n = 0; n < width_kernel_conv_CNN; n++) {
						sum += weight[m * width_kernel_conv_CNN + n] * image_input[m * width_image_input_CNN + n];
					}
				}

				image[y * width_image_C1_CNN + x] = activation_function_tanh(sum + bias_C1[i]); //tanh((w*x + b))
			}
		}
	}*/

	for (int o = 0; o < num_map_C1_CNN; o++) {
		for (int inc = 0; inc < num_map_input_CNN; inc++) {
			int addr1 = get_index(0, 0, num_map_input_CNN * o + inc, width_kernel_conv_CNN, height_kernel_conv_CNN, num_map_C1_CNN);
			int addr2 = get_index(0, 0, inc, width_image_input_CNN, height_image_input_CNN, num_map_input_CNN);
			int addr3 = get_index(0, 0, o, width_image_C1_CNN, height_image_C1_CNN, num_map_C1_CNN);

			const float* pw = &weight_C1[0] + addr1;
			const float* pi = data_single_image + addr2;
			float* pa = &neuron_C1[0] + addr3;

			for (int y = 0; y < height_image_C1_CNN; y++) {
				for (int x = 0; x < width_image_C1_CNN; x++) {
					const float* ppw = pw;
					const float* ppi = pi + y * width_image_input_CNN + x;
					float sum = 0.0;

					for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
						for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
							sum += *ppw++ * ppi[wy * width_image_input_CNN + wx];
						}
					}

					pa[y * width_image_C1_CNN + x] += sum;
				}
			}
		}

		int addr3 = get_index(0, 0, o, width_image_C1_CNN, height_image_C1_CNN, num_map_C1_CNN);
		float* pa = &neuron_C1[0] + addr3;
		float b = bias_C1[o];
		for (int y = 0; y < height_image_C1_CNN; y++) {
			for (int x = 0; x < width_image_C1_CNN; x++) {
				pa[y * width_image_C1_CNN + x] += b;
			}
		}
	}

	for (int i = 0; i < num_neuron_C1_CNN; i++) {
		neuron_C1[i] = activation_function_tanh(neuron_C1[i]);
	}

	return true;
}

void CNN::calc_out2wi(int width_in, int height_in, int width_out, int height_out, int depth_out, std::vector<wi_connections>& out2wi)
{
	for (int i = 0; i < depth_out; i++) {
		int block = width_in * height_in * i;

		for (int y = 0; y < height_out; y++) {
			for (int x = 0; x < width_out; x++) {
				int rows = y * width_kernel_pooling_CNN;
				int cols = x * height_kernel_pooling_CNN;

				wi_connections wi_connections_;
				std::pair<int, int> pair_;

				for (int m = 0; m < width_kernel_pooling_CNN; m++) {
					for (int n = 0; n < height_kernel_pooling_CNN; n++) {
						pair_.first = i;
						pair_.second = (rows + m) * width_in + cols + n + block;
						wi_connections_.push_back(pair_);
					}
				}
				out2wi.push_back(wi_connections_);
			}
		}
	}
}

void CNN::calc_out2bias(int width, int height, int depth, std::vector<int>& out2bias)
{
	for (int i = 0; i < depth; i++) {
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				out2bias.push_back(i);
			}
		}
	}
}

void CNN::calc_in2wo(int width_in, int height_in, int width_out, int height_out, int depth_in, int depth_out, std::vector<wo_connections>& in2wo)
{
	int len = width_in * height_in * depth_in;
	in2wo.resize(len);

	for (int c = 0; c < depth_in; c++) {
		for (int y = 0; y < height_in; y += height_kernel_pooling_CNN) {
			for (int x = 0; x < width_in; x += width_kernel_pooling_CNN) {
				int dymax = min(size_pooling_CNN, height_in - y);
				int dxmax = min(size_pooling_CNN, width_in - x);
				int dstx = x / width_kernel_pooling_CNN;
				int dsty = y / height_kernel_pooling_CNN;

				for (int dy = 0; dy < dymax; dy++) {
					for (int dx = 0; dx < dxmax; dx++) {
						int index_in = get_index(x + dx, y + dy, c, width_in, height_in, depth_in);
						int index_out = get_index(dstx, dsty, c, width_out, height_out, depth_out);

						wo_connections wo_connections_;
						std::pair<int, int> pair_;
						pair_.first = c;
						pair_.second = index_out;
						wo_connections_.push_back(pair_);

						in2wo[index_in] = wo_connections_;
					}
				}
			}
		}
	}
}

void CNN::calc_weight2io(int width_in, int height_in, int width_out, int height_out, int depth_in, int depth_out, std::vector<io_connections>& weight2io)
{
	int len = depth_in;
	weight2io.resize(len);

	for (int c = 0; c < depth_in; c++) {
		for (int y = 0; y < height_in; y += height_kernel_pooling_CNN) {
			for (int x = 0; x < width_in; x += width_kernel_pooling_CNN) {
				int dymax = min(size_pooling_CNN, height_in - y);
				int dxmax = min(size_pooling_CNN, width_in - x);
				int dstx = x / width_kernel_pooling_CNN;
				int dsty = y / height_kernel_pooling_CNN;

				for (int dy = 0; dy < dymax; dy++) {
					for (int dx = 0; dx < dxmax; dx++) {
						int index_in = get_index(x + dx, y + dy, c, width_in, height_in, depth_in);
						int index_out = get_index(dstx, dsty, c, width_out, height_out, depth_out);

						std::pair<int, int> pair_;
						pair_.first = index_in;
						pair_.second = index_out;

						weight2io[c].push_back(pair_);
					}
				}
			}
		}
	}
}

void CNN::calc_bias2out(int width_in, int height_in, int width_out, int height_out, int depth_in, int depth_out, std::vector<std::vector<int> >& bias2out)
{
	int len = depth_in;
	bias2out.resize(len);

	for (int c = 0; c < depth_in; c++) {
		for (int y = 0; y < height_out; y++) {
			for (int x = 0; x < width_out; x++) {
				int index_out = get_index(x, y, c, width_out, height_out, depth_out);
				bias2out[c].push_back(index_out);
			}
		}
	}
}

bool CNN::Forward_S2()
{
	init_variable(neuron_S2, 0.0, num_neuron_S2_CNN);
	float scale_factor = 1.0 / (width_kernel_pooling_CNN * height_kernel_pooling_CNN);

	/*for (int i = 0; i < num_map_S2_CNN; i++) {
		int addr1 = i * width_image_S2_CNN * height_image_S2_CNN;
		int addr2 = i * width_image_C1_CNN * height_image_C1_CNN;

		float* image = &neuron_S2[0] + addr1;
		const float* image_input = &neuron_C1[0] + addr2;

		for (int y = 0; y < height_image_S2_CNN; y++) {
			for (int x = 0; x < width_image_S2_CNN; x++) {
				float sum = 0.0;
				int rows = y * height_kernel_pooling_CNN;
				int cols = x * width_kernel_pooling_CNN;

				for (int m = 0; m < height_kernel_pooling_CNN; m++) {
					for (int n = 0; n < width_kernel_pooling_CNN; n++) {
						sum += image_input[(rows + m) * width_image_C1_CNN + cols + n];
					}
				}

				image[y * width_image_S2_CNN + x] = activation_function_tanh(sum * weight_S2[i] * scale_factor + bias_S2[i]);
			}
		}
	}*/

	assert(out2wi_S2.size() == num_neuron_S2_CNN);
	assert(out2bias_S2.size() == num_neuron_S2_CNN);

	for (int i = 0; i < num_neuron_S2_CNN; i++) {
		const wi_connections& connections = out2wi_S2[i];
		neuron_S2[i] = 0;

		for (int index = 0; index < connections.size(); index++) {
			neuron_S2[i] += weight_S2[connections[index].first] * neuron_C1[connections[index].second];
		}

		neuron_S2[i] *= scale_factor;
		neuron_S2[i] += bias_S2[out2bias_S2[i]];
	}

	for (int i = 0; i < num_neuron_S2_CNN; i++) {
		neuron_S2[i] = activation_function_tanh(neuron_S2[i]);
	}

	return true;
}

bool CNN::Forward_C3()
{
	init_variable(neuron_C3, 0.0, num_neuron_C3_CNN);

	/*for (int i = 0; i < num_map_C3_CNN; i++) {
		int addr1 = i * width_image_C3_CNN * height_image_C3_CNN;
		int addr2 = i * width_kernel_conv_CNN * height_kernel_conv_CNN * num_map_S2_CNN;
		float* image = &neuron_C3[0] + addr1;
		const float* weight = &weight_C3[0] + addr2;

		for (int j = 0; j < num_map_S2_CNN; j++) {
			int addr3 = j * width_image_S2_CNN * height_image_S2_CNN;
			int addr4 = j * width_kernel_conv_CNN * height_kernel_conv_CNN;
			const float* image_input = &neuron_S2[0] + addr3;
			const float* weight_ = weight + addr4;

			for (int y = 0; y < height_image_C3_CNN; y++) {
				for (int x = 0; x < width_image_C3_CNN; x++) {
					float sum = 0.0;
					const float* image_input_ = image_input + y * width_image_S2_CNN + x;

					for (int m = 0; m < height_kernel_conv_CNN; m++) {
						for (int n = 0; n < width_kernel_conv_CNN; n++) {
							sum += weight_[m * width_kernel_conv_CNN + n] * image_input_[m * width_image_S2_CNN + n];
						}
					}

					image[y * width_image_C3_CNN + x] += sum;
				}
			}
		}

		for (int y = 0; y < height_image_C3_CNN; y++) {
			for (int x = 0; x < width_image_C3_CNN; x++) {
				image[y * width_image_C3_CNN + x] = activation_function_tanh(image[y * width_image_C3_CNN + x] + bias_C3[i]);
			}
		}
	}*/

	for (int o = 0; o < num_map_C3_CNN; o++) {
		for (int inc = 0; inc < num_map_S2_CNN; inc++) {
			int addr1 = get_index(0, 0, num_map_S2_CNN * o + inc, width_kernel_conv_CNN, height_kernel_conv_CNN, num_map_C3_CNN * num_map_S2_CNN);
			int addr2 = get_index(0, 0, inc, width_image_S2_CNN, height_image_S2_CNN, num_map_S2_CNN);
			int addr3 = get_index(0, 0, o, width_image_C3_CNN, height_image_C3_CNN, num_map_C3_CNN);

			const float* pw = &weight_C3[0] + addr1;
			const float* pi = &neuron_S2[0] + addr2;
			float* pa = &neuron_C3[0] + addr3;

			for (int y = 0; y < height_image_C3_CNN; y++) {
				for (int x = 0; x < width_image_C3_CNN; x++) {
					const float* ppw = pw;
					const float* ppi = pi + y * width_image_S2_CNN + x;
					float sum = 0.0;

					for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
						for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
							sum += *ppw++ * ppi[wy * width_image_S2_CNN + wx];
						}
					}

					pa[y * width_image_C3_CNN + x] += sum;
				}
			}
		}

		int addr3 = get_index(0, 0, o, width_image_C3_CNN, height_image_C3_CNN, num_map_C3_CNN);
		float* pa = &neuron_C3[0] + addr3;
		float b = bias_C3[o];
		for (int y = 0; y < height_image_C3_CNN; y++) {
			for (int x = 0; x < width_image_C3_CNN; x++) {
				pa[y * width_image_C3_CNN + x] += b;
			}
		}
	}

	for (int i = 0; i < num_neuron_C3_CNN; i++) {
		neuron_C3[i] = activation_function_tanh(neuron_C3[i]);
	}

	return true;
}

bool CNN::Forward_S4()
{
	float scale_factor = 1.0 / (width_kernel_pooling_CNN * height_kernel_pooling_CNN);
	init_variable(neuron_S4, 0.0, num_neuron_S4_CNN);

	/*for (int i = 0; i < num_map_S4_CNN; i++) {
		int addr1 = i * width_image_S4_CNN * height_image_S4_CNN;
		int addr2 = i * width_image_C3_CNN * height_image_C3_CNN;

		float* image = &neuron_S4[0] + addr1;
		const float* image_input = &neuron_C3[0] + addr2;

		for (int y = 0; y < height_image_S4_CNN; y++) {
			for (int x = 0; x < width_image_S4_CNN; x++) {
				float sum = 0.0;
				int rows = y * height_kernel_pooling_CNN;
				int cols = x * width_kernel_pooling_CNN;

				for (int m = 0; m < height_kernel_pooling_CNN; m++) {
					for (int n = 0; n < width_kernel_pooling_CNN; n++) {
						sum += image_input[(rows + m) * width_image_C3_CNN + cols + n];
					}
				}

				image[y * width_image_S4_CNN + x] = activation_function_tanh(sum * weight_S4[i] * scale_factor + bias_S4[i]);
			}
		}
	}*/

	assert(out2wi_S4.size() == num_neuron_S4_CNN);
	assert(out2bias_S4.size() == num_neuron_S4_CNN);

	for (int i = 0; i < num_neuron_S4_CNN; i++) {
		const wi_connections& connections = out2wi_S4[i];
		neuron_S4[i] = 0.0;

		for (int index = 0; index < connections.size(); index++) {
			neuron_S4[i] += weight_S4[connections[index].first] * neuron_C3[connections[index].second];
		}

		neuron_S4[i] *= scale_factor;
		neuron_S4[i] += bias_S4[out2bias_S4[i]];
	}

	for (int i = 0; i < num_neuron_S4_CNN; i++) {
		neuron_S4[i] = activation_function_tanh(neuron_S4[i]);
	}

	//int count_num = 0;
	//for (int i = 0; i < num_neuron_S4_CNN; i++) {
	//	if (fabs(neuron_S4[i] - Tmp_neuron_S4[i]) > 0.0000001/*0.0000000001*/) {
	//		count_num++;
	//		std::cout << "i = " << i << " , old: " << neuron_S4[i] << " , new: " << Tmp_neuron_S4[i] << std::endl;
	//	}
	//}
	//std::cout << "count_num: " << count_num << std::endl;

	return true;
}

bool CNN::Forward_C5()
{
	init_variable(neuron_C5, 0.0, num_neuron_C5_CNN);

	/*for (int i = 0; i < num_map_C5_CNN; i++) {
		int addr1 = i * width_image_C5_CNN * height_image_C5_CNN;
		int addr2 = i * width_kernel_conv_CNN * height_kernel_conv_CNN * num_map_S4_CNN;
		float* image = &neuron_C5[0] + addr1;
		const float* weight = &weight_C5[0] + addr2;

		for (int j = 0; j < num_map_S4_CNN; j++) {
			int addr3 = j * width_kernel_conv_CNN * height_kernel_conv_CNN;
			int addr4 = j * width_image_S4_CNN * height_image_S4_CNN;
			const float* weight_ = weight + addr3;
			const float* image_input = &neuron_S4[0] + addr4;

			for (int y = 0; y < height_image_C5_CNN; y++) {
				for (int x = 0; x < width_image_C5_CNN; x++) {
					float sum = 0.0;
					const float* image_input_ = image_input + y * width_image_S4_CNN + x;

					for (int m = 0; m < height_kernel_conv_CNN; m++) {
						for (int n = 0; n < width_kernel_conv_CNN; n++) {
							sum += weight_[m * width_kernel_conv_CNN + n] * image_input_[m * width_image_S4_CNN + n];
						}
					}

					image[y * width_image_C5_CNN + x] += sum;
				}
			}
		}

		for (int y = 0; y < height_image_C5_CNN; y++) {
			for (int x = 0; x < width_image_C5_CNN; x++) {
				image[y * width_image_C5_CNN + x] = activation_function_tanh(image[y * width_image_C5_CNN + x] + bias_C5[i]);
			}
		}
	}*/

	for (int o = 0; o < num_map_C5_CNN; o++) {
		for (int inc = 0; inc < num_map_S4_CNN; inc++) {
			int addr1 = get_index(0, 0, num_map_S4_CNN * o + inc, width_kernel_conv_CNN, height_kernel_conv_CNN, num_map_C5_CNN * num_map_S4_CNN);
			int addr2 = get_index(0, 0, inc, width_image_S4_CNN, height_image_S4_CNN, num_map_S4_CNN);
			int addr3 = get_index(0, 0, o, width_image_C5_CNN, height_image_C5_CNN, num_map_C5_CNN);

			const float *pw = &weight_C5[0] + addr1;
			const float *pi = &neuron_S4[0] + addr2;
			float *pa = &neuron_C5[0] + addr3;

			for (int y = 0; y < height_image_C5_CNN; y++) {
				for (int x = 0; x < width_image_C5_CNN; x++) {
					const float *ppw = pw;
					const float *ppi = pi + y * width_image_S4_CNN + x;
					float sum = 0.0;

					for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
						for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
							sum += *ppw++ * ppi[wy * width_image_S4_CNN + wx];
						}
					}

					pa[y * width_image_C5_CNN + x] += sum;
				}
			}
		}

		int addr3 = get_index(0, 0, o, width_image_C5_CNN, height_image_C5_CNN, num_map_C5_CNN);
		float *pa = &neuron_C5[0] + addr3;
		float b = bias_C5[o];
		for (int y = 0; y < height_image_C5_CNN; y++) {
			for (int x = 0; x < width_image_C5_CNN; x++) {
				pa[y * width_image_C5_CNN + x] += b;
			}
		}
	}

	for (int i = 0; i < num_neuron_C5_CNN; i++) {
		neuron_C5[i] = activation_function_tanh(neuron_C5[i]);
	}

	return true;
}

bool CNN::Forward_output()
{
	init_variable(neuron_output, 0.0, num_neuron_output_CNN);
	/*float* image = &neuron_output[0];
	const float* weight = &weight_output[0];

	for (int i = 0; i < num_neuron_output_CNN; i++) {
		for (int j = 0; j < num_neuron_C5_CNN; j++) {
			image[i] += (weight[j * num_neuron_output_CNN + i] * neuron_C5[j]);
		}

		image[i] = activation_function_tanh(image[i] + bias_output[i]);
	}*/

	for (int i = 0; i < num_neuron_output_CNN; i++) {
		neuron_output[i] = 0.0;

		for (int c = 0; c < num_neuron_C5_CNN; c++) {
			neuron_output[i] += weight_output[c * num_neuron_output_CNN + i] * neuron_C5[c];
		}

		neuron_output[i] += bias_output[i];
	}

	for (int i = 0; i < num_neuron_output_CNN; i++) {
		neuron_output[i] = activation_function_tanh(neuron_output[i]);
	}

	return true;
}

bool CNN::Backward_output()
{
	init_variable(delta_neuron_output, 0.0, num_neuron_output_CNN);
	/*float gradient[num_neuron_output_CNN];
	const float* t = &data_single_label[0];
	float tmp[num_neuron_output_CNN];

	for (int i = 0; i < num_neuron_output_CNN; i++) {
		gradient[i] = loss_function_mse_derivative(neuron_output[i], t[i]);
	}

	for (int i = 0; i < num_neuron_output_CNN; i++) {
		init_variable(tmp, 0.0, num_neuron_output_CNN);
		tmp[i] = activation_function_tanh_derivative(neuron_output[i]);

		delta_neuron_output[i] = dot_product(gradient, tmp, num_neuron_output_CNN);
	}*/

	float dE_dy[num_neuron_output_CNN];
	init_variable(dE_dy, 0.0, num_neuron_output_CNN);
	loss_function_gradient(neuron_output, data_single_label, dE_dy, num_neuron_output_CNN);
	
	// delta = dE/da = (dE/dy) * (dy/da)
	for (int i = 0; i < num_neuron_output_CNN; i++) {
		float dy_da[num_neuron_output_CNN];
		init_variable(dy_da, 0.0, num_neuron_output_CNN);

		dy_da[i] = activation_function_tanh_derivative(neuron_output[i]);
		delta_neuron_output[i] = dot_product(dE_dy, dy_da, num_neuron_output_CNN);
	}

	return true;
}

bool CNN::Backward_C5()
{
	init_variable(delta_neuron_C5, 0.0, num_neuron_C5_CNN);
	init_variable(delta_weight_output, 0.0, len_weight_output_CNN);
	init_variable(delta_bias_output, 0.0, len_bias_output_CNN);

	/*for (int i = 0; i < num_neuron_C5_CNN; i++) {
		delta_neuron_C5[i] = dot_product(&delta_neuron_output[0], &weight_output[0] + i * num_neuron_output_CNN, num_neuron_output_CNN);
		delta_neuron_C5[i] *= activation_function_tanh_derivative(neuron_C5[i]);
	}

	for (int j = 0; j < num_neuron_C5_CNN; j++) {
		muladd(&delta_neuron_output[0], neuron_C5[j], num_neuron_output_CNN, &delta_weight_output[0] + j * num_neuron_output_CNN);
	}

	for (int i = 0; i < num_neuron_output_CNN; i++) {
		delta_bias_output[i] += delta_neuron_output[i];
	}*/

	for (int c = 0; c < num_neuron_C5_CNN; c++) {
		// propagate delta to previous layer
		// prev_delta[c] += current_delta[r] * W_[c * out_size_ + r]
		delta_neuron_C5[c] = dot_product(&delta_neuron_output[0], &weight_output[c * num_neuron_output_CNN], num_neuron_output_CNN);
		delta_neuron_C5[c] *= activation_function_tanh_derivative(neuron_C5[c]);
	}

	// accumulate weight-step using delta
	// dW[c * out_size + i] += current_delta[i] * prev_out[c]
	for (int c = 0; c < num_neuron_C5_CNN; c++) {
		muladd(&delta_neuron_output[0], neuron_C5[c], num_neuron_output_CNN, &delta_weight_output[0] + c * num_neuron_output_CNN);
	}

	for (int i = 0; i < len_bias_output_CNN; i++) {
		delta_bias_output[i] += delta_neuron_output[i];
	}

	//int count_num = 0;
	//for (int i = 0; i < num_neuron_C5_CNN; i++) {
	//	if (fabs(delta_neuron_C5[i] - Tmp_delta_neuron_C5[i]) > 0.0000001/*0.0000000001*/) {
	//		count_num++;
	//	}
	//}
	//std::cout << "delta_neuron count_num: " << count_num << std::endl;
	//count_num = 0;
	//for (int i = 0; i < len_weight_output_CNN; i++) {
	//	if (fabs(delta_weight_output[i] - Tmp_delta_weight_output[i]) > 0.0000001/*0.0000000001*/) {
	//		count_num++;
	//	}
	//}
	//std::cout << "delta_weight count_num: " << count_num << std::endl;
	//count_num = 0;
	//for (int i = 0; i < len_bias_output_CNN; i++) {
	//	if (fabs(delta_bias_output[i] - Tmp_delta_bias_output[i]) > 0.0000001/*0.0000000001*/) {
	//		count_num++;
	//	}
	//}
	//std::cout << "delta_bias count_num: " << count_num << std::endl;

	return true;
}

bool CNN::Backward_S4()
{
	init_variable(delta_neuron_S4, 0.0, num_neuron_S4_CNN);
	init_variable(delta_weight_C5, 0.0, len_weight_C5_CNN);
	init_variable(delta_bias_C5, 0.0, len_bias_C5_CNN);

	/*for (int i = 0; i < num_map_S4_CNN; i++) {
		for (int j = 0; j < num_map_C5_CNN; j++) {
			int addr1 = width_kernel_conv_CNN * height_kernel_conv_CNN * (num_map_S4_CNN * j + i);
			int addr2 = width_image_S4_CNN * height_image_S4_CNN * i;

			const float* weight_c5 = &weight_C5[0] + addr1;
			const float* delta_c5 = &delta_neuron_C5[0] + width_image_C5_CNN * height_image_C5_CNN * j;
			float* delta_s4 = &delta_neuron_S4[0] + addr2;

			for (int y = 0; y < height_image_C5_CNN; y++) {
				for (int x = 0; x < width_image_C5_CNN; x++) {
					const float* weight_c5_ = weight_c5;
					const float delta_c5_ = delta_c5[y * width_image_C5_CNN + x];
					float* delta_s4_ = delta_s4 + y * width_image_S4_CNN + x;

					for (int m = 0; m < height_kernel_conv_CNN; m++) {
						for (int n = 0; n < width_kernel_conv_CNN; n++) {
							delta_s4_[m * width_image_S4_CNN + n] += weight_c5_[m * width_kernel_conv_CNN + n] * delta_c5_;
						}
					}
				}
			}
		}
	}

	for (int i = 0; i < num_neuron_S4_CNN; i++) {
		delta_neuron_S4[i] *= activation_function_tanh_derivative(neuron_S4[i]);
	}

	for (int i = 0; i < num_map_S4_CNN; i++) {////////
		for (int j = 0; j < num_map_C5_CNN; j++) {
			for (int y = 0; y < height_kernel_conv_CNN; y++) {
				for (int x = 0; x < width_kernel_conv_CNN; x++) {
					int addr1 = (height_image_S4_CNN * i + y) * width_image_S4_CNN + x;
					int addr2 = (height_kernel_conv_CNN * (num_map_S4_CNN * j + i) + y) * width_kernel_conv_CNN + x;
					int addr3 = height_image_C5_CNN * j * width_image_C5_CNN;

					float dst = 0;
					const float* neuron_s4 = &neuron_S4[0] + addr1;
					const float* delta_c5 = &delta_neuron_C5[0] + addr3;

					for (int m = 0; m < height_image_C5_CNN; m++) {
						dst += dot_product(neuron_s4 + m * width_image_S4_CNN, delta_c5 + y * width_image_C5_CNN, width_image_C5_CNN);
					}

					delta_weight_C5[addr2] += dst;
				}
			}
		}
	}

	for (int i = 0; i < num_map_C5_CNN; i++) {
		delta_bias_C5[i] += delta_neuron_C5[i];
	}*/

	// propagate delta to previous layer
	for (int inc = 0; inc < num_map_S4_CNN; inc++) {
		for (int outc = 0; outc < num_map_C5_CNN; outc++) {
			int addr1 = get_index(0, 0, num_map_S4_CNN * outc + inc, width_kernel_conv_CNN, height_kernel_conv_CNN, num_map_S4_CNN * num_map_C5_CNN);
			int addr2 = get_index(0, 0, outc, width_image_C5_CNN, height_image_C5_CNN, num_map_C5_CNN);
			int addr3 = get_index(0, 0, inc, width_image_S4_CNN, height_image_S4_CNN, num_map_S4_CNN);

			const float* pw = &weight_C5[0] + addr1;
			const float* pdelta_src = &delta_neuron_C5[0] + addr2;
			float* pdelta_dst = &delta_neuron_S4[0] + addr3;

			for (int y = 0; y < height_image_C5_CNN; y++) {
				for (int x = 0; x < width_image_C5_CNN; x++) {
					const float* ppw = pw;
					const float ppdelta_src = pdelta_src[y * width_image_C5_CNN + x];
					float* ppdelta_dst = pdelta_dst + y * width_image_S4_CNN + x;

					for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
						for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
							ppdelta_dst[wy * width_image_S4_CNN + wx] += *ppw++ * ppdelta_src;
						}
					}
				}
			}
		}
	}

	for (int i = 0; i < num_neuron_S4_CNN; i++) {
		delta_neuron_S4[i] *= activation_function_tanh_derivative(neuron_S4[i]);
	}

	// accumulate dw
	for (int inc = 0; inc < num_map_S4_CNN; inc++) {
		for (int outc = 0; outc < num_map_C5_CNN; outc++) {
			for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
				for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
					int addr1 = get_index(wx, wy, inc, width_image_S4_CNN, height_image_S4_CNN, num_map_S4_CNN);
					int addr2 = get_index(0, 0, outc, width_image_C5_CNN, height_image_C5_CNN, num_map_C5_CNN);
					int addr3 = get_index(wx, wy, num_map_S4_CNN * outc + inc, width_kernel_conv_CNN, height_kernel_conv_CNN, num_map_S4_CNN * num_map_C5_CNN);

					float dst = 0.0;
					const float* prevo = &neuron_S4[0] + addr1;
					const float* delta = &delta_neuron_C5[0] + addr2;

					for (int y = 0; y < height_image_C5_CNN; y++) {
						dst += dot_product(prevo + y * width_image_S4_CNN, delta + y * width_image_C5_CNN, width_image_C5_CNN);
					}

					delta_weight_C5[addr3] += dst;
				}
			}
		}
	}

	// accumulate db
	for (int outc = 0; outc < num_map_C5_CNN; outc++) {
		int addr2 = get_index(0, 0, outc, width_image_C5_CNN, height_image_C5_CNN, num_map_C5_CNN);
		const float* delta = &delta_neuron_C5[0] + addr2;

		for (int y = 0; y < height_image_C5_CNN; y++) {
			for (int x = 0; x < width_image_C5_CNN; x++) {
				delta_bias_C5[outc] += delta[y * width_image_C5_CNN + x];
			}
		}
	}

	return true;
}

bool CNN::Backward_C3()
{
	init_variable(delta_neuron_C3, 0.0, num_neuron_C3_CNN);
	init_variable(delta_weight_S4, 0.0, len_weight_S4_CNN);
	init_variable(delta_bias_S4, 0.0, len_bias_S4_CNN);

	float scale_factor = 1.0 / (width_kernel_pooling_CNN * height_kernel_pooling_CNN);

	/*for (int i = 0; i < num_map_C3_CNN; i++) {
		int addr1 = width_image_S4_CNN * height_image_S4_CNN * i;
		int addr2 = width_image_C3_CNN * height_image_C3_CNN * i;

		const float* delta_s4 = &delta_neuron_S4[0] + addr1;
		float* delta_c3 = &delta_neuron_C3[0] + addr2;
		const float* neuron_c3 = &neuron_C3[0] + addr2;

		for (int y = 0; y < height_image_C3_CNN; y++) {
			for (int x = 0; x < width_image_C3_CNN; x++) {
				float delta = 0.0;
				int index = width_image_S4_CNN * (y / height_kernel_pooling_CNN) + x / width_kernel_pooling_CNN;
				delta = weight_S4[i] * delta_s4[index];

				delta_c3[y * width_image_C3_CNN + x] = delta * scale_factor * activation_function_tanh_derivative(neuron_c3[y * width_image_C3_CNN + x]);
			}
		}
	}

	for (int i = 0; i < len_weight_S4_CNN; i++) {
		int addr1 = width_image_C3_CNN * height_image_C3_CNN * i;
		int addr2 = width_image_S4_CNN * height_image_S4_CNN * i;

		const float* neuron_c3 = &neuron_C3[0] + addr1;
		const float* delta_s4 = &delta_neuron_S4[0] + addr2;

		float diff = 0.0;

		for (int y = 0; y < height_image_C3_CNN; y++) {
			for (int x = 0; x < width_image_C3_CNN; x++) {
				int index = y / height_kernel_pooling_CNN * height_image_S4_CNN + x / width_kernel_pooling_CNN;

				diff += neuron_c3[y * width_image_C3_CNN + x] * delta_s4[index];
			}
		}

		delta_weight_S4[i] += diff * scale_factor;
	}

	for (int i = 0; i < len_bias_S4_CNN; i++) {
		int addr1 = width_image_S4_CNN * height_image_S4_CNN * i;
		const float* delta_s4 = &delta_neuron_S4[0] + addr1;
		float diff = 0;

		for (int y = 0; y < height_image_S4_CNN; y++) {
			for (int x = 0; x < width_image_S4_CNN; x++) {
				diff += delta_s4[y * width_image_S4_CNN + x];
			}
		}

		delta_bias_S4[i] += diff;
	}*/

	assert(in2wo_C3.size() == num_neuron_C3_CNN);
	assert(weight2io_C3.size() == len_weight_S4_CNN);
	assert(bias2out_C3.size() == len_bias_S4_CNN);

	for (int i = 0; i < num_neuron_C3_CNN; i++) {
		const wo_connections& connections = in2wo_C3[i];
		float delta = 0.0;

		for (int j = 0; j < connections.size(); j++) {
			delta += weight_S4[connections[j].first] * delta_neuron_S4[connections[j].second];
		}

		delta_neuron_C3[i] = delta * scale_factor * activation_function_tanh_derivative(neuron_C3[i]);
	}

	for (int i = 0; i < len_weight_S4_CNN; i++) {
		const io_connections& connections = weight2io_C3[i];
		float diff = 0;

		for (int j = 0; j < connections.size(); j++) {
			diff += neuron_C3[connections[j].first] * delta_neuron_S4[connections[j].second];
		}

		delta_weight_S4[i] += diff * scale_factor;
	}

	for (int i = 0; i < len_bias_S4_CNN; i++) {
		const std::vector<int>& outs = bias2out_C3[i];
		float diff = 0;

		for (int o = 0; o < outs.size(); o++) {
			diff += delta_neuron_S4[outs[o]];
		}

		delta_bias_S4[i] += diff;
	}

	return true;
}

bool CNN::Backward_S2()
{
	init_variable(delta_neuron_S2, 0.0, num_neuron_S2_CNN);
	init_variable(delta_weight_C3, 0.0, len_weight_C3_CNN);
	init_variable(delta_bias_C3, 0.0, len_bias_C3_CNN);

	/*for (int i = 0; i < num_map_S2_CNN; i++) {////////////////
		int addr1 = width_kernel_conv_CNN * height_kernel_conv_CNN * num_map_C3_CNN * i;
		int addr2 = width_kernel_conv_CNN * height_kernel_conv_CNN * i;
		for (int j = 0; j < num_map_C3_CNN; j++) {
			const float* weight_c3 = &weight_C3[0] + addr1 + j * width_kernel_conv_CNN * height_kernel_conv_CNN;
			const float* delta_c3 = &delta_neuron_C3[0] + width_image_C3_CNN * height_image_C3_CNN * j;
			float* delta_s2 = &delta_neuron_S2[0] + addr2;

			for (int y = 0; y < height_image_C3_CNN; y++) {
				for (int x = 0; x < width_image_C3_CNN; x++) {
					const float* weight_c3_ = weight_c3;
					const float delta_c3_ = delta_c3[y * width_image_C3_CNN + x];
					float* delta_s2_ = delta_s2 + y * width_kernel_conv_CNN + x;

					for (int m = 0; m < height_kernel_conv_CNN; m++) {
						for (int n = 0; n < width_kernel_conv_CNN; n++) {
							delta_s2_[m * width_kernel_conv_CNN + n] += weight_c3_[m * width_kernel_conv_CNN + n] * delta_c3_;
						}
					}
				}
			}
		}
	}

	for (int i = 0; i < num_neuron_S2_CNN; i++) {
		delta_neuron_S2[i] *= activation_function_tanh_derivative(neuron_S2[i]);
	}

	for (int i = 0; i < num_map_S2_CNN; i++) {//////////////////
		int addr1 = width_kernel_conv_CNN * height_kernel_conv_CNN * i;

		for (int j = 0; j < num_map_C3_CNN; j++) {
			int addr2 = width_kernel_conv_CNN * height_kernel_conv_CNN * i * j;
			float* delta_weight_c3 = &delta_weight_C3[0] + addr2;

			for (int y = 0; y < height_kernel_conv_CNN; y++) {
				for (int x = 0; x < width_kernel_conv_CNN; x++) {
					float dst = 0;
					const float* neuron_s2 = &neuron_S2[0] + addr1 + y * width_kernel_conv_CNN + x;
					const float* delta_c3 = &delta_neuron_C3[0] + width_image_C3_CNN * height_image_C3_CNN * j;

					for (int m = 0; m < height_image_C3_CNN; m++) {
						dst += dot_product(neuron_s2 + m * width_kernel_conv_CNN, delta_c3 + y * width_image_C3_CNN, width_image_C3_CNN);
					}

					delta_weight_c3[y * width_kernel_conv_CNN + x] += dst;
				}
			}
		}
	}

	for (int i = 0; i < num_map_C3_CNN; i++) {
		const float* delta = &delta_neuron_C3[0] + width_image_C3_CNN * height_image_C3_CNN * i;

		//delta_bias_C3[i] += std::accumulate(delta, delta + width_image_C3_CNN * height_image_C3_CNN, (float)0.0);
		for (int y = 0; y < height_image_C3_CNN; y++) {
			for (int x = 0; x < width_image_C3_CNN; x++) {
				delta_bias_C3[i] += delta[y * width_image_C3_CNN + x];
			}
		}
	}*/

	// propagate delta to previous layer
	for (int inc = 0; inc < num_map_S2_CNN; inc++) {
		for (int outc = 0; outc < num_map_C3_CNN; outc++) {
			int addr1 = get_index(0, 0, num_map_S2_CNN * outc + inc, width_kernel_conv_CNN, height_kernel_conv_CNN, num_map_S2_CNN * num_map_C3_CNN);
			int addr2 = get_index(0, 0, outc, width_image_C3_CNN, height_image_C3_CNN, num_map_C3_CNN);
			int addr3 = get_index(0, 0, inc, width_image_S2_CNN, height_image_S2_CNN, num_map_S2_CNN);

			const float *pw = &weight_C3[0] + addr1;
			const float *pdelta_src = &delta_neuron_C3[0] + addr2;;
			float* pdelta_dst = &delta_neuron_S2[0] + addr3;

			for (int y = 0; y < height_image_C3_CNN; y++) {
				for (int x = 0; x < width_image_C3_CNN; x++) {
					const float* ppw = pw;
					const float ppdelta_src = pdelta_src[y * width_image_C3_CNN + x];
					float* ppdelta_dst = pdelta_dst + y * width_image_S2_CNN + x;

					for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
						for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
							ppdelta_dst[wy * width_image_S2_CNN + wx] += *ppw++ * ppdelta_src;
						}
					}
				}
			}
		}
	}

	for (int i = 0; i < num_neuron_S2_CNN; i++) {
		delta_neuron_S2[i] *= activation_function_tanh_derivative(neuron_S2[i]);
	}

	// accumulate dw
	for (int inc = 0; inc < num_map_S2_CNN; inc++) {
		for (int outc = 0; outc < num_map_C3_CNN; outc++) {
			for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
				for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
					int addr1 = get_index(wx, wy, inc, width_image_S2_CNN, height_image_S2_CNN, num_map_S2_CNN);
					int addr2 = get_index(0, 0, outc, width_image_C3_CNN, height_image_C3_CNN, num_map_C3_CNN);
					int addr3 = get_index(wx, wy, num_map_S2_CNN * outc + inc, width_kernel_conv_CNN, height_kernel_conv_CNN, num_map_S2_CNN * num_map_C3_CNN);
					
					float dst = 0.0;
					const float* prevo = &neuron_S2[0] + addr1;
					const float* delta = &delta_neuron_C3[0] + addr2;

					for (int y = 0; y < height_image_C3_CNN; y++) {
						dst += dot_product(prevo + y * width_image_S2_CNN, delta + y * width_image_C3_CNN, width_image_C3_CNN);
					}

					delta_weight_C3[addr3] += dst;
				}
			}
		}
	}

	// accumulate db
	for (int outc = 0; outc < len_bias_C3_CNN; outc++) {
		int addr1 = get_index(0, 0, outc, width_image_C3_CNN, height_image_C3_CNN, num_map_C3_CNN);
		const float* delta = &delta_neuron_C3[0] + addr1;

		for (int y = 0; y < height_image_C3_CNN; y++) {
			for (int x = 0; x < width_image_C3_CNN; x++) {
				delta_bias_C3[outc] += delta[y * width_image_C3_CNN + x];
			}
		}
	}

	return true;
}

bool CNN::Backward_C1()
{
	init_variable(delta_neuron_C1, 0.0, num_neuron_C1_CNN);
	init_variable(delta_weight_S2, 0.0, len_weight_S2_CNN);
	init_variable(delta_bias_S2, 0.0, len_bias_S2_CNN);

	float scale_factor = 1.0 / (width_kernel_pooling_CNN * height_kernel_pooling_CNN);

	/*for (int i = 0; i < num_map_C1_CNN; i++) {
		int addr1 = width_image_S2_CNN * height_image_S2_CNN * i;
		int addr2 = width_image_C1_CNN * height_image_C1_CNN * i;

		const float* delta_s2 = &delta_neuron_S2[0] + addr1;
		float* delta_c1 = &delta_neuron_C1[0] + addr2;
		const float* neuron_c1 = &neuron_C1[0] + addr2;

		for (int y = 0; y < height_image_C1_CNN; y++) {
			for (int x = 0; x < width_image_C1_CNN; x++) {
				float delta = 0.0;
				int index = width_image_S2_CNN * (y / height_kernel_pooling_CNN) + x / width_kernel_pooling_CNN;
				delta = weight_S2[i] * delta_s2[index];

				delta_c1[y * width_image_C1_CNN + x] = delta * scale_factor * activation_function_tanh_derivative(neuron_c1[y * width_image_C1_CNN + x]);
			}
		}
	}

	for (int i = 0; i < len_weight_S2_CNN; i++) {
		int addr1 = width_image_C1_CNN * height_image_C1_CNN * i;
		int addr2 = width_image_S2_CNN * height_image_S2_CNN * i;

		const float* neuron_c1 = &neuron_C1[0] + addr1;
		const float* delta_s2 = &delta_neuron_S2[0] + addr2;

		float diff = 0.0;

		for (int y = 0; y < height_image_C1_CNN; y++) {
			for (int x = 0; x < width_image_C1_CNN; x++) {
				int index = y / height_kernel_pooling_CNN * height_image_S2_CNN + x / width_kernel_pooling_CNN;

				diff += neuron_c1[y * width_image_C1_CNN + x] * delta_s2[index];
			}
		}

		delta_weight_S2[i] += diff * scale_factor;
	}

	for (int i = 0; i < len_bias_S2_CNN; i++) {
		int addr1 = width_image_S2_CNN * height_image_S2_CNN * i;
		const float* delta_s2 = &delta_neuron_S2[0] + addr1;
		float diff = 0;

		for (int y = 0; y < height_image_S2_CNN; y++) {
			for (int x = 0; x < width_image_S2_CNN; x++) {
				diff += delta_s2[y * width_image_S2_CNN + x];
			}
		}

		delta_bias_S2[i] += diff;
	}*/

	assert(in2wo_C1.size() == num_neuron_C1_CNN);
	assert(weight2io_C1.size() == len_weight_S2_CNN);
	assert(bias2out_C1.size() == len_bias_S2_CNN);

	for (int i = 0; i < num_neuron_C1_CNN; i++) {
		const wo_connections& connections = in2wo_C1[i];
		float delta = 0.0;

		for (int j = 0; j < connections.size(); j++) {
			delta += weight_S2[connections[j].first] * delta_neuron_S2[connections[j].second];
		}

		delta_neuron_C1[i] = delta * scale_factor * activation_function_tanh_derivative(neuron_C1[i]);
	}

	for (int i = 0; i < len_weight_S2_CNN; i++) {
		const io_connections& connections = weight2io_C1[i];
		float diff = 0.0;

		for (int j = 0; j < connections.size(); j++) {
			diff += neuron_C1[connections[j].first] * delta_neuron_S2[connections[j].second];
		}

		delta_weight_S2[i] += diff * scale_factor;
	}

	for (int i = 0; i < len_bias_S2_CNN; i++) {
		const std::vector<int>& outs = bias2out_C1[i];
		float diff = 0;

		for (int o = 0; o < outs.size(); o++) {
			diff += delta_neuron_S2[outs[o]];
		}

		delta_bias_S2[i] += diff;
	}

	return true;
}

bool CNN::Backward_input()
{
	init_variable(delta_neuron_input, 0.0, num_neuron_input_CNN);
	init_variable(delta_weight_C1, 0.0, len_weight_C1_CNN);
	init_variable(delta_bias_C1, 0.0, len_bias_C1_CNN);

	/*for (int i = 0; i < num_map_input_CNN; i++) {///////////////////
		int addr1 = width_kernel_conv_CNN * height_kernel_conv_CNN * num_map_C1_CNN * i;
		int addr2 = width_image_input_CNN * height_image_input_CNN * i;
		for (int j = 0; j < num_map_C1_CNN; j++) {
			const float* weight_c1 = &weight_C1[0] + addr1 + j * width_kernel_conv_CNN * height_kernel_conv_CNN;
			const float* delta_c1 = &delta_neuron_C1[0] + width_image_C1_CNN * height_image_C1_CNN * j;
			float* delta_input_ = &delta_neuron_input[0] + addr2;

			for (int y = 0; y < height_image_C1_CNN; y++) {
				for (int x = 0; x < width_image_C1_CNN; x++) {
					const float* weight_c1_ = weight_c1;
					const float delta_c1_ = delta_c1[y * width_image_C1_CNN + x];
					float* delta_input_0 = delta_input_ + y * width_image_C1_CNN + x;

					for (int m = 0; m < height_kernel_conv_CNN; m++) {
						for (int n = 0; n < width_kernel_conv_CNN; n++) {
							delta_input_0[m * width_image_input_CNN + n] += weight_c1_[m * width_kernel_conv_CNN + n] * delta_c1_;
						}
					}
				}
			}
		}
	}

	for (int i = 0; i < num_neuron_input_CNN; i++) {
		delta_neuron_input[i] *= activation_function_identity_derivative(data_single_image[i]);
	}

	for (int i = 0; i < num_map_input_CNN; i++) {/////////////
		int addr1 = width_image_input_CNN * height_image_input_CNN * i;

		for (int j = 0; j < num_map_C1_CNN; j++) {
			int addr2 = width_kernel_conv_CNN * height_kernel_conv_CNN * i * j;
			float* delta_weight_c1 = &delta_weight_C1[0] + addr2;

			for (int y = 0; y < height_kernel_conv_CNN; y++) {
				for (int x = 0; x < width_kernel_conv_CNN; x++) {
					float dst = 0;
					const float* neuron_input_ = data_single_image + addr1 + y * width_image_input_CNN + x;
					const float* delta_c1 = &delta_neuron_C1[0] + width_image_C1_CNN * height_image_C1_CNN * j;

					for (int m = 0; m < height_image_C1_CNN; m++) {
						dst += dot_product(neuron_input_ + m * width_kernel_conv_CNN, delta_c1 + y * width_image_C1_CNN, width_image_C1_CNN);
					}

					delta_weight_c1[y * width_kernel_conv_CNN + x] += dst;
				}
			}
		}
	}

	for (int i = 0; i < num_map_C1_CNN; i++) {
		const float* delta = &delta_neuron_C1[0] + width_image_C1_CNN * height_image_C1_CNN * i;

		//delta_bias_C1[i] += std::accumulate(delta, delta + width_image_C1_CNN * height_image_C1_CNN, (float)0.0);
		for (int y = 0; y < height_image_C1_CNN; y++) {
			for (int x = 0; x < width_image_C1_CNN; x++) {
				delta_bias_C1[i] += delta[y * width_image_C1_CNN + x];
			}
		}
	}*/

	// propagate delta to previous layer
	for (int inc = 0; inc < num_map_input_CNN; inc++) {
		for (int outc = 0; outc < num_map_C1_CNN; outc++) {
			int addr1 = get_index(0, 0, num_map_input_CNN * outc + inc, width_kernel_conv_CNN, height_kernel_conv_CNN, num_map_C1_CNN);
			int addr2 = get_index(0, 0, outc, width_image_C1_CNN, height_image_C1_CNN, num_map_C1_CNN);
			int addr3 = get_index(0, 0, inc, width_image_input_CNN, height_image_input_CNN, num_map_input_CNN);

			const float* pw = &weight_C1[0] + addr1;
			const float* pdelta_src = &delta_neuron_C1[0] + addr2;
			float* pdelta_dst = &delta_neuron_input[0] + addr3;

			for (int y = 0; y < height_image_C1_CNN; y++) {
				for (int x = 0; x < width_image_C1_CNN; x++) {
					const float* ppw = pw;
					const float ppdelta_src = pdelta_src[y * width_image_C1_CNN + x];
					float* ppdelta_dst = pdelta_dst + y * width_image_input_CNN + x;

					for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
						for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
							ppdelta_dst[wy * width_image_input_CNN + wx] += *ppw++ * ppdelta_src;
						}
					}
				}
			}
		}
	}

	for (int i = 0; i < num_neuron_input_CNN; i++) {
		delta_neuron_input[i] *= activation_function_identity_derivative(data_single_image[i]/*neuron_input[i]*/);
	}

	// accumulate dw
	for (int inc = 0; inc < num_map_input_CNN; inc++) {
		for (int outc = 0; outc < num_map_C1_CNN; outc++) {
			for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
				for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
					int addr1 = get_index(wx, wy, inc, width_image_input_CNN, height_image_input_CNN, num_map_input_CNN);
					int addr2 = get_index(0, 0, outc, width_image_C1_CNN, height_image_C1_CNN, num_map_C1_CNN);
					int addr3 = get_index(wx, wy, num_map_input_CNN * outc + inc, width_kernel_conv_CNN, height_kernel_conv_CNN, num_map_C1_CNN);

					float dst = 0.0;
					const float* prevo = data_single_image + addr1;//&neuron_input[0]
					const float* delta = &delta_neuron_C1[0] + addr2;

					for (int y = 0; y < height_image_C1_CNN; y++) {
						dst += dot_product(prevo + y * width_image_input_CNN, delta + y * width_image_C1_CNN, width_image_C1_CNN);
					}

					delta_weight_C1[addr3] += dst;
				}
			}
		}
	}

	// accumulate db
	for (int outc = 0; outc < len_bias_C1_CNN; outc++) {
		int addr1 = get_index(0, 0, outc, width_image_C1_CNN, height_image_C1_CNN, num_map_C1_CNN);
		const float* delta = &delta_neuron_C1[0] + addr1;

		for (int y = 0; y < height_image_C1_CNN; y++) {
			for (int x = 0; x < width_image_C1_CNN; x++) {
				delta_bias_C1[outc] += delta[y * width_image_C1_CNN + x];
			}
		}
	}

	return true;
}

void CNN::update_weights_bias(const float* delta, float* weight, int len)
{
	for (int i = 0; i < len; i++) {
		float tmp = delta[i] * delta[i];
		weight[i] -= learning_rate_CNN * delta[i] / (std::sqrt(tmp) + eps_CNN);
	}
}

bool CNN::UpdateWeights()
{
	update_weights_bias(delta_weight_C1, weight_C1, len_weight_C1_CNN);
	update_weights_bias(delta_bias_C1, bias_C1, len_bias_C1_CNN);

	update_weights_bias(delta_weight_S2, weight_S2, len_weight_S2_CNN);
	update_weights_bias(delta_bias_S2, bias_S2, len_bias_S2_CNN);

	update_weights_bias(delta_weight_C3, weight_C3, len_weight_C3_CNN);
	update_weights_bias(delta_bias_C3, bias_C3, len_bias_C3_CNN);

	update_weights_bias(delta_weight_S4, weight_S4, len_weight_S4_CNN);
	update_weights_bias(delta_bias_S4, bias_S4, len_bias_S4_CNN);

	update_weights_bias(delta_weight_C5, weight_C5, len_weight_C5_CNN);
	update_weights_bias(delta_bias_C5, bias_C5, len_bias_C5_CNN);

	update_weights_bias(delta_weight_output, weight_output, len_weight_output_CNN);
	update_weights_bias(delta_bias_output, bias_output, len_bias_output_CNN);

	return true;
}

int CNN::predict(const unsigned char* data, int width, int height)
{
	assert(data && width == width_image_input_CNN && height == height_image_input_CNN);

	const float scale_min = -1;
	const float scale_max = 1;

	float tmp[width_image_input_CNN * height_image_input_CNN];
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			tmp[y * width + x] = (data[y * width + x] / 255.0) * (scale_max - scale_min) + scale_min;
		}
	}

	data_single_image = &tmp[0];

	Forward_C1();
	Forward_S2();
	Forward_C3();
	Forward_S4();
	Forward_C5();
	Forward_output();

	int pos = -1;
	float max_value = -9999.0;

	for (int i = 0; i < num_neuron_output_CNN; i++) {
		if (neuron_output[i] > max_value) {
			max_value = neuron_output[i];
			pos = i;
		}
	}

	return pos;
}

bool CNN::readModelFile(const char* name)
{
	FILE* fp = fopen(name, "rb");
	if (fp == NULL) {
		return false;
	}

	int width_image_input =0;
	int height_image_input = 0;
	int width_image_C1 = 0;
	int height_image_C1 = 0;
	int width_image_S2 = 0;
	int height_image_S2 = 0;
	int width_image_C3 = 0;
	int height_image_C3 = 0;
	int width_image_S4 = 0;
	int height_image_S4 = 0;
	int width_image_C5 = 0;
	int height_image_C5 = 0;
	int width_image_output = 0;
	int height_image_output = 0;

	int width_kernel_conv = 0;
	int height_kernel_conv = 0;
	int width_kernel_pooling = 0;
	int height_kernel_pooling = 0;

	int num_map_input = 0;
	int num_map_C1 = 0;
	int num_map_S2 = 0;
	int num_map_C3 = 0;
	int num_map_S4 = 0;
	int num_map_C5 = 0;
	int num_map_output = 0;

	int len_weight_C1 = 0;
	int len_bias_C1 = 0;
	int len_weight_S2 = 0;
	int len_bias_S2 = 0;
	int len_weight_C3 = 0;
	int len_bias_C3 = 0;
	int len_weight_S4 = 0;
	int len_bias_S4 = 0;
	int len_weight_C5 = 0;
	int len_bias_C5 = 0;
	int len_weight_output = 0;
	int len_bias_output = 0;

	int num_neuron_input = 0;
	int num_neuron_C1 = 0;
	int num_neuron_S2 = 0;
	int num_neuron_C3 = 0;
	int num_neuron_S4 = 0;
	int num_neuron_C5 = 0;
	int num_neuron_output = 0;

	fread(&width_image_input, sizeof(int), 1, fp);
	fread(&height_image_input, sizeof(int), 1, fp);
	fread(&width_image_C1, sizeof(int), 1, fp);
	fread(&height_image_C1, sizeof(int), 1, fp);
	fread(&width_image_S2, sizeof(int), 1, fp);
	fread(&height_image_S2, sizeof(int), 1, fp);
	fread(&width_image_C3, sizeof(int), 1, fp);
	fread(&height_image_C3, sizeof(int), 1, fp);
	fread(&width_image_S4, sizeof(int), 1, fp);
	fread(&height_image_S4, sizeof(int), 1, fp);
	fread(&width_image_C5, sizeof(int), 1, fp);
	fread(&height_image_C5, sizeof(int), 1, fp);
	fread(&width_image_output, sizeof(int), 1, fp);
	fread(&height_image_output, sizeof(int), 1, fp);

	fread(&width_kernel_conv, sizeof(int), 1, fp);
	fread(&height_kernel_conv, sizeof(int), 1, fp);
	fread(&width_kernel_pooling, sizeof(int), 1, fp);
	fread(&height_kernel_pooling, sizeof(int), 1, fp);

	fread(&num_map_input, sizeof(int), 1, fp);
	fread(&num_map_C1, sizeof(int), 1, fp);
	fread(&num_map_S2, sizeof(int), 1, fp);
	fread(&num_map_C3, sizeof(int), 1, fp);
	fread(&num_map_S4, sizeof(int), 1, fp);
	fread(&num_map_C5, sizeof(int), 1, fp);
	fread(&num_map_output, sizeof(int), 1, fp);

	fread(&len_weight_C1, sizeof(int), 1, fp);
	fread(&len_bias_C1, sizeof(int), 1, fp);
	fread(&len_weight_S2, sizeof(int), 1, fp);
	fread(&len_bias_S2, sizeof(int), 1, fp);
	fread(&len_weight_C3, sizeof(int), 1, fp);
	fread(&len_bias_C3, sizeof(int), 1, fp);
	fread(&len_weight_S4, sizeof(int), 1, fp);
	fread(&len_bias_S4, sizeof(int), 1, fp);
	fread(&len_weight_C5, sizeof(int), 1, fp);
	fread(&len_bias_C5, sizeof(int), 1, fp);
	fread(&len_weight_output, sizeof(int), 1, fp);
	fread(&len_bias_output, sizeof(int), 1, fp);

	fread(&num_neuron_input, sizeof(int), 1, fp);
	fread(&num_neuron_C1, sizeof(int), 1, fp);
	fread(&num_neuron_S2, sizeof(int), 1, fp);
	fread(&num_neuron_C3, sizeof(int), 1, fp);
	fread(&num_neuron_S4, sizeof(int), 1, fp);
	fread(&num_neuron_C5, sizeof(int), 1, fp);
	fread(&num_neuron_output, sizeof(int), 1, fp);

	fread(weight_C1, sizeof(weight_C1), 1, fp);
	fread(bias_C1, sizeof(bias_C1), 1, fp);
	fread(weight_S2, sizeof(weight_S2), 1, fp);
	fread(bias_S2, sizeof(bias_S2), 1, fp);
	fread(weight_C3, sizeof(weight_C3), 1, fp);
	fread(bias_C3, sizeof(bias_C3), 1, fp);
	fread(weight_S4, sizeof(weight_S4), 1, fp);
	fread(bias_S4, sizeof(bias_S4), 1, fp);
	fread(weight_C5, sizeof(weight_C5), 1, fp);
	fread(bias_C5, sizeof(bias_C5), 1, fp);
	fread(weight_output, sizeof(weight_output), 1, fp);
	fread(bias_output, sizeof(bias_output), 1, fp);

	fflush(fp);
	fclose(fp);

	out2wi_S2.clear();
	out2bias_S2.clear();
	out2wi_S4.clear();
	out2bias_S4.clear();

	calc_out2wi(width_image_C1_CNN, height_image_C1_CNN, width_image_S2_CNN, height_image_S2_CNN, num_map_S2_CNN, out2wi_S2);
	calc_out2bias(width_image_S2_CNN, height_image_S2_CNN, num_map_S2_CNN, out2bias_S2);
	calc_out2wi(width_image_C3_CNN, height_image_C3_CNN, width_image_S4_CNN, height_image_S4_CNN, num_map_S4_CNN, out2wi_S4);
	calc_out2bias(width_image_S4_CNN, height_image_S4_CNN, num_map_S4_CNN, out2bias_S4);

	return true;
}

bool CNN::saveModelFile(const char* name)
{
	FILE* fp = fopen(name, "wb");
	if (fp == NULL) {
		return false;
	}

	int width_image_input = width_image_input_CNN;
	int height_image_input = height_image_input_CNN;
	int width_image_C1 = width_image_C1_CNN;
	int height_image_C1 = height_image_C1_CNN;
	int width_image_S2 = width_image_S2_CNN;
	int height_image_S2 = height_image_S2_CNN;
	int width_image_C3 = width_image_C3_CNN;
	int height_image_C3 = height_image_C3_CNN;
	int width_image_S4 = width_image_S4_CNN;
	int height_image_S4 = height_image_S4_CNN;
	int width_image_C5 = width_image_C5_CNN;
	int height_image_C5 = height_image_C5_CNN;
	int width_image_output = width_image_output_CNN;
	int height_image_output = height_image_output_CNN;

	int width_kernel_conv = width_kernel_conv_CNN;
	int height_kernel_conv = height_kernel_conv_CNN;
	int width_kernel_pooling = width_kernel_pooling_CNN;
	int height_kernel_pooling = height_kernel_pooling_CNN;

	int num_map_input = num_map_input_CNN;
	int num_map_C1 = num_map_C1_CNN;
	int num_map_S2 = num_map_S2_CNN;
	int num_map_C3 = num_map_C3_CNN;
	int num_map_S4 = num_map_S4_CNN;
	int num_map_C5 = num_map_C5_CNN;
	int num_map_output = num_map_output_CNN;

	int len_weight_C1 = len_weight_C1_CNN;
	int len_bias_C1 = len_bias_C1_CNN;
	int len_weight_S2 = len_weight_S2_CNN;
	int len_bias_S2 = len_bias_S2_CNN;
	int len_weight_C3 = len_weight_C3_CNN;
	int len_bias_C3 = len_bias_C3_CNN;
	int len_weight_S4 = len_weight_S4_CNN;
	int len_bias_S4 = len_bias_S4_CNN;
	int len_weight_C5 = len_weight_C5_CNN;
	int len_bias_C5 = len_bias_C5_CNN;
	int len_weight_output = len_weight_output_CNN;
	int len_bias_output = len_bias_output_CNN;

	int num_neuron_input = num_neuron_input_CNN;
	int num_neuron_C1 = num_neuron_C1_CNN;
	int num_neuron_S2 = num_neuron_S2_CNN;
	int num_neuron_C3 = num_neuron_C3_CNN;
	int num_neuron_S4 = num_neuron_S4_CNN;
	int num_neuron_C5 = num_neuron_C5_CNN;
	int num_neuron_output = num_neuron_output_CNN;

	fwrite(&width_image_input, sizeof(int), 1, fp);
	fwrite(&height_image_input, sizeof(int), 1, fp);
	fwrite(&width_image_C1, sizeof(int), 1, fp);
	fwrite(&height_image_C1, sizeof(int), 1, fp);
	fwrite(&width_image_S2, sizeof(int), 1, fp);
	fwrite(&height_image_S2, sizeof(int), 1, fp);
	fwrite(&width_image_C3, sizeof(int), 1, fp);
	fwrite(&height_image_C3, sizeof(int), 1, fp);
	fwrite(&width_image_S4, sizeof(int), 1, fp);
	fwrite(&height_image_S4, sizeof(int), 1, fp);
	fwrite(&width_image_C5, sizeof(int), 1, fp);
	fwrite(&height_image_C5, sizeof(int), 1, fp);
	fwrite(&width_image_output, sizeof(int), 1, fp);
	fwrite(&height_image_output, sizeof(int), 1, fp);

	fwrite(&width_kernel_conv, sizeof(int), 1, fp);
	fwrite(&height_kernel_conv, sizeof(int), 1, fp);
	fwrite(&width_kernel_pooling, sizeof(int), 1, fp);
	fwrite(&height_kernel_pooling, sizeof(int), 1, fp);

	fwrite(&num_map_input, sizeof(int), 1, fp);
	fwrite(&num_map_C1, sizeof(int), 1, fp);
	fwrite(&num_map_S2, sizeof(int), 1, fp);
	fwrite(&num_map_C3, sizeof(int), 1, fp);
	fwrite(&num_map_S4, sizeof(int), 1, fp);
	fwrite(&num_map_C5, sizeof(int), 1, fp);
	fwrite(&num_map_output, sizeof(int), 1, fp);

	fwrite(&len_weight_C1, sizeof(int), 1, fp);
	fwrite(&len_bias_C1, sizeof(int), 1, fp);
	fwrite(&len_weight_S2, sizeof(int), 1, fp);
	fwrite(&len_bias_S2, sizeof(int), 1, fp);
	fwrite(&len_weight_C3, sizeof(int), 1, fp);
	fwrite(&len_bias_C3, sizeof(int), 1, fp);
	fwrite(&len_weight_S4, sizeof(int), 1, fp);
	fwrite(&len_bias_S4, sizeof(int), 1, fp);
	fwrite(&len_weight_C5, sizeof(int), 1, fp);
	fwrite(&len_bias_C5, sizeof(int), 1, fp);
	fwrite(&len_weight_output, sizeof(int), 1, fp);
	fwrite(&len_bias_output, sizeof(int), 1, fp);

	fwrite(&num_neuron_input, sizeof(int), 1, fp);
	fwrite(&num_neuron_C1, sizeof(int), 1, fp);
	fwrite(&num_neuron_S2, sizeof(int), 1, fp);
	fwrite(&num_neuron_C3, sizeof(int), 1, fp);
	fwrite(&num_neuron_S4, sizeof(int), 1, fp);
	fwrite(&num_neuron_C5, sizeof(int), 1, fp);
	fwrite(&num_neuron_output, sizeof(int), 1, fp);

	fwrite(weight_C1, sizeof(weight_C1), 1, fp);
	fwrite(bias_C1, sizeof(bias_C1), 1, fp);
	fwrite(weight_S2, sizeof(weight_S2), 1, fp);
	fwrite(bias_S2, sizeof(bias_S2), 1, fp);
	fwrite(weight_C3, sizeof(weight_C3), 1, fp);
	fwrite(bias_C3, sizeof(bias_C3), 1, fp);
	fwrite(weight_S4, sizeof(weight_S4), 1, fp);
	fwrite(bias_S4, sizeof(bias_S4), 1, fp);
	fwrite(weight_C5, sizeof(weight_C5), 1, fp);
	fwrite(bias_C5, sizeof(bias_C5), 1, fp);
	fwrite(weight_output, sizeof(weight_output), 1, fp);
	fwrite(bias_output, sizeof(bias_output), 1, fp);

	fflush(fp);
	fclose(fp);

	return true;
}

float CNN::test()
{
	int count_accuracy = 0;

	for (int num = 0; num < num_patterns_test_CNN; num++) {
		data_single_image = data_input_test + num * num_neuron_input_CNN;
		data_single_label = data_output_test + num * num_neuron_output_CNN;

		Forward_C1();
		Forward_S2();
		Forward_C3();
		Forward_S4();
		Forward_C5();
		Forward_output();

		int pos_t = -1;
		int pos_y = -2;
		float max_value_t = -9999.0;
		float max_value_y = -9999.0;

		for (int i = 0; i < num_neuron_output_CNN; i++) {
			if (neuron_output[i] > max_value_y) {
				max_value_y = neuron_output[i];
				pos_y = i;
			}

			if (data_single_label[i] > max_value_t) {
				max_value_t = data_single_label[i];
				pos_t = i;
			}
		}

		if (pos_y == pos_t) {
			++count_accuracy;
		}

		Sleep(1);
	}

	//std::cout << "count_accuracy: " << count_accuracy << std::endl;
	return (count_accuracy * 1.0 / num_patterns_test_CNN);
}

}
