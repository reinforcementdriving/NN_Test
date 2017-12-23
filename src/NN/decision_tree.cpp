#include "decision_tree.hpp"
#include "common.hpp"

namespace ANN {

template<typename T>
int DecisionTree<T>::init(const std::vector<std::vector<T>>& data, const std::vector<T>& classes)
{
	CHECK(data.size() != 0 && classes.size() != 0 && data[0].size() != 0);

	this->samples_num = data.size();
	this->classes_num = classes.size();
	this->feature_length = data[0].size() -1;

	this->features.resize(this->samples_num);
	this->labels.resize(this->samples_num);
	for (int i = 0; i < this->samples_num; ++i) {
		this->labels[i] = data[i][this->feature_length];

		this->features[i].resize(this->feature_length);
		for (int j = 0; j < this->feature_length; ++j) {
			this->features[i][j] = data[i][j];
		}
	}

	/*int count = 0;
	std::vector<std::vector<std::vector<T>>> groups(2);
	for (int i = 0; i < 2; ++i) {
		groups[i].resize(3);
		for (int j = 0; j < 3; ++j) {
			groups[i][j].resize(this->feature_length + 1);
			for (int t = 0; t < this->feature_length + 1; ++t) {
				groups[i][j][t] = data[count][t];
			}
			++count;
		}
	}*/

	//T gini = gini_index(groups, classes);
	//fprintf(stderr, "gini: %f\n", gini);
	dictionary dic = build_tree(data, 1, 1);
	fprintf(stderr, "Split: [X%d < %.3f]\n", std::get<0>(dic)+1, std::get<1>(dic));

	return 0;
}

template<typename T>
T DecisionTree<T>::gini_index(const std::vector<std::vector<std::vector<T>>>& groups, const std::vector<T>& classes) const
{
	// count all samples at split point
	int instances = 0;
	int group_num = groups.size();
	for (int i = 0; i < group_num; ++i) {
		instances += groups[i].size();
	}

	// sum weighted Gini index for each group
	T gini = (T)0.;
	for (int i = 0; i < group_num; ++i) {
		int size = groups[i].size();
		// avoid divide by zero
		if (size == 0) continue;
		T score = (T)0.;

		// score the group based on the score for each class
		T p = (T)0.;
		for (int c = 0; c < classes.size(); ++c) {
			int count = 0;
			for (int t = 0; t < size; ++t) {
				if (groups[i][t][this->feature_length] == classes[c]) ++count;
			}
			T p = (float)count / size;
			score += p * p;
		}

		// weight the group score by its relative size
		gini += (1. - score) * (float)size / instances;
	}

	return gini;
}

template<typename T>
std::vector<std::vector<std::vector<T>>> DecisionTree<T>::test_split(int index, T value, const std::vector<std::vector<T>>& dataset) const
{
	std::vector<std::vector<std::vector<T>>> groups(2); // 0: left, 1: reight

	for (int row = 0; row < dataset.size(); ++row) {
		if (dataset[row][index] < value) {
			groups[0].emplace_back(dataset[row]);
		} else {
			groups[1].emplace_back(dataset[row]);
		}
	}

	return groups;
}

template<typename T>
std::tuple<int, T, std::vector<std::vector<std::vector<T>>>> DecisionTree<T>::get_split(const std::vector<std::vector<T>>& dataset) const
{
	std::vector<T> values;
	for (int i = 0; i < dataset.size(); ++i) {
		values.emplace_back(dataset[i][this->feature_length]);
	}

	std::vector<T> class_values = get_unique_value(values);

	int b_index = 999;
	T b_value = (T)999.;
	T b_score = (T)999.;
	std::vector<std::vector<std::vector<T>>> b_groups;

	for (int index = 0; index < this->feature_length; ++index) {
		for (int row = 0; row < dataset.size(); ++row) {
			std::vector<std::vector<std::vector<T>>> groups = test_split(index, dataset[row][index], dataset);
			T gini = gini_index(groups, class_values);
			//fprintf(stderr, "X%d < %.3f Gini = %.3f\n", index + 1, dataset[row][index], gini);

			if (gini < b_score) {
				b_index = index;
				b_value = dataset[row][index];
				b_score = gini;
				b_groups = groups;
			}
		}
	}

	return std::make_tuple(b_index, b_value, b_groups);
}

template<typename T>
T DecisionTree<T>::to_terminal(const std::vector<std::vector<T>>& group) const
{
	std::vector<T> values;
	for (int i = 0; i < group.size(); ++i) {
		values.emplace_back(group[i][this->feature_length]);
	}

	std::vector<T> class_values = get_unique_value(values);
	std::vector<int> class_count(class_values.size(), 0);

	for (int i = 0; i < values.size(); ++i) {
		for (int j = 0; j < class_values.size(); ++j) {
			if (values[i] == class_values[j]) {
				++class_count[j];
				break;
			}
		}
	}

	int max_count{ class_count[0] }, max_count_index{ 0 };
	for (int i = 1; i < class_count.size(); ++i) {
		if (max_count < class_count[i]) {
			max_count_index = i;
			max_count = class_count[i];
		}
	}

	return class_values[max_count_index];
}

template<typename T>
std::vector<T> DecisionTree<T>::get_unique_value(const std::vector<T>& values) const
{
	if (values.size() == 0) return std::vector<T>();

	std::vector<T> class_values;
	class_values.emplace_back(values[0]);

	for (int i = 1; i < values.size(); ++i) {
		int j = 0;
		for (; j < class_values.size(); ++j) {
			if (values[i] == class_values[j]) {
				break;
			}
		}

		if (j == class_values.size()) {
			class_values.emplace_back(values[i]);
		}
	}

	return class_values;
}

template<typename T>
void DecisionTree<T>::split(dictionary& node, int max_depth, int min_size, int depth) const
{
	std::vector<std::vector<T>> left = std::get<2>(node)[0]; // left
	std::vector<std::vector<T>> right = std::get<2>(node)[1]; // right

	// check for a no split
	// TODO
}

template<typename T>
std::tuple<int, T, std::vector<std::vector<std::vector<T>>>> DecisionTree<T>::build_tree(const std::vector<std::vector<T>>& train, int max_depth, int min_size) const
{
	dictionary root = get_split(train);
	split(root, max_depth, min_size, 1);

	return root;
}

template<typename T>
void DecisionTree<T>::print_tree(const dictionary& node, int depth) const
{

}

template class DecisionTree<float>;
template class DecisionTree<double>;

} // namespace ANN

