#include "funset.hpp"
#include <string>
#include <algorithm>
#include "tiny_dnn/tiny_dnn.h"

static void construct_net(tiny_dnn::network<tiny_dnn::sequential>& nn)
{
	// connection table [Y.Lecun, 1998 Table.1]
#define O true
#define X false
	static const bool tbl[] = {
		O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
		O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
		O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
		X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
		X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
		X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
	};
#undef O
#undef X

	// by default will use backend_t::tiny_dnn unless you compiled
	// with -DUSE_AVX=ON and your device supports AVX intrinsics
	tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();

	// construct nets: C: convolution; S: sub-sampling; F: fully connected
	nn << tiny_dnn::convolutional_layer<tiny_dnn::activation::tan_h>(32, 32, 5, 1, 6,  // C1, 1@32x32-in, 6@28x28-out
		tiny_dnn::padding::valid, true, 1, 1, backend_type)
		<< tiny_dnn::average_pooling_layer<tiny_dnn::activation::tan_h>(28, 28, 6, 2)   // S2, 6@28x28-in, 6@14x14-out
		<< tiny_dnn::convolutional_layer<tiny_dnn::activation::tan_h>(14, 14, 5, 6, 16, // C3, 6@14x14-in, 16@10x10-out
		connection_table(tbl, 6, 16),
		tiny_dnn::padding::valid, true, 1, 1, backend_type)
		<< tiny_dnn::average_pooling_layer<tiny_dnn::activation::tan_h>(10, 10, 16, 2)  // S4, 16@10x10-in, 16@5x5-out
		<< tiny_dnn::convolutional_layer<tiny_dnn::activation::tan_h>(5, 5, 5, 16, 120, // C5, 16@5x5-in, 120@1x1-out
		tiny_dnn::padding::valid, true, 1, 1, backend_type)
		<< tiny_dnn::fully_connected_layer<tiny_dnn::activation::tan_h>(120, 10,        // F6, 120-in, 10-out
		true, backend_type);
}

static void train_lenet(const std::string& data_dir_path)
{
	// specify loss-function and learning strategy
	tiny_dnn::network<tiny_dnn::sequential> nn;
	tiny_dnn::adagrad optimizer;

	construct_net(nn);

	std::cout << "load models..." << std::endl;

	// load MNIST dataset
	std::vector<tiny_dnn::label_t> train_labels, test_labels;
	std::vector<tiny_dnn::vec_t> train_images, test_images;

	tiny_dnn::parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte", &train_labels);
	tiny_dnn::parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte", &train_images, -1.0, 1.0, 2, 2);
	tiny_dnn::parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte", &test_labels);
	tiny_dnn::parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte", &test_images, -1.0, 1.0, 2, 2);

	std::cout << "start training" << std::endl;

	tiny_dnn::progress_display disp(static_cast<unsigned long>(train_images.size()));
	tiny_dnn::timer t;
	int minibatch_size = 10;
	int num_epochs = 30;

	optimizer.alpha *= static_cast<tiny_dnn::float_t>(std::sqrt(minibatch_size));

	// create callback
	auto on_enumerate_epoch = [&](){
		std::cout << t.elapsed() << "s elapsed." << std::endl;
		tiny_dnn::result res = nn.test(test_images, test_labels);
		std::cout << res.num_success << "/" << res.num_total << std::endl;

		disp.restart(static_cast<unsigned long>(train_images.size()));
		t.restart();
	};

	auto on_enumerate_minibatch = [&](){
		disp += minibatch_size;
	};

	// training
	nn.train<tiny_dnn::mse>(optimizer, train_images, train_labels, minibatch_size, num_epochs, on_enumerate_minibatch, on_enumerate_epoch);

	std::cout << "end training." << std::endl;

	// test and show results
	nn.test(test_images, test_labels).print_detail(std::cout);

	// save network model & trained weights
	nn.save(data_dir_path + "/LeNet-model");
}

// rescale output to 0-100
template <typename Activation>
static double rescale(double x)
{
	Activation a;
	return 100.0 * (x - a.scale().first) / (a.scale().second - a.scale().first);
}

static void convert_image(const std::string& imagefilename, double minv, double maxv, int w, int h, tiny_dnn::vec_t& data)
{
	tiny_dnn::image<> img(imagefilename, tiny_dnn::image_type::grayscale);
	tiny_dnn::image<> resized = resize_image(img, w, h);

	// mnist dataset is "white on black", so negate required
	std::transform(resized.begin(), resized.end(), std::back_inserter(data),
		[=](uint8_t c) { return (255 - c) * (maxv - minv) / 255.0 + minv; });
}

int test_dnn_mnist_train()
{
	std::string data_dir_path = "E:/GitCode/NN_Test/data/database/MNIST";
	train_lenet(data_dir_path);

	return 0;
}

int test_dnn_mnist_predict()
{
	std::string model { "E:/GitCode/NN_Test/data/LeNet-model" };
	std::string image_path { "E:/GitCode/NN_Test/data/images/digit/handwriting_2/"};
	int target[10] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

	tiny_dnn::network<tiny_dnn::sequential> nn;
	nn.load(model);

	for (int i = 0; i < 10; i++) {
		std::string str = std::to_string(i);
		str += ".png";
		str = image_path + str;

		// convert imagefile to vec_t
		tiny_dnn::vec_t data;
		convert_image(str, -1.0, 1.0, 32, 32, data);

		// recognize
		auto res = nn.predict(data);
		std::vector<std::pair<double, int> > scores;

		// sort & print top-3
		for (int j = 0; j < 10; j++)
			scores.emplace_back(rescale<tiny_dnn::tan_h>(res[j]), j);

		std::sort(scores.begin(), scores.end(), std::greater<std::pair<double, int>>());

		for (int j = 0; j < 3; j++)
			fprintf(stdout, "%d: %f;  ", scores[j].second, scores[j].first);
		fprintf(stderr, "\n");

		// save outputs of each layer
		for (size_t j = 0; j < nn.depth(); j++) {
			auto out_img = nn[j]->output_to_image();
			auto filename = image_path + std::to_string(i) + "_layer_" + std::to_string(j) + ".png";
			out_img.save(filename);
		}

		// save filter shape of first convolutional layer
		auto weight = nn.at<tiny_dnn::convolutional_layer<tiny_dnn::tan_h>>(0).weight_to_image();
		auto filename = image_path + std::to_string(i) + "_weights.png";
		weight.save(filename);

		fprintf(stdout, "the actual digit is: %d, correct digit is: %d \n\n", scores[0].second, target[i]);
	}

	return 0;
}
