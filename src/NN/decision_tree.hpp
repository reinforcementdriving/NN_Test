#ifndef FBC_NN_DECISION_TREE_HPP_
#define FBC_NN_DECISION_TREE_HPP_

#include <vector>
#include <tuple>

namespace ANN {
// referecne: https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/

template<typename T>
class DecisionTree {
public:
	typedef std::tuple<int, T, std::vector<std::vector<std::vector<T>>>> dictionary; // index, value, groups
	DecisionTree() = default;
	int init(const std::vector<std::vector<T>>& data, const std::vector<T>& classes);

protected:
	// Calculate the Gini index for a split dataset
	T gini_index(const std::vector<std::vector<std::vector<T>>>& groups, const std::vector<T>& classes) const;
	// Select the best split point for a dataset
	dictionary get_split(const std::vector<std::vector<T>>& dataset) const;
	// Split a dataset based on an attribute and an attribute value
	std::vector<std::vector<std::vector<T>>> test_split(int index, T value, const std::vector<std::vector<T>>& dataset) const;
	// Create a terminal node value
	T to_terminal(const std::vector<std::vector<T>>& group) const;
	std::vector<T> get_unique_value(const std::vector<T>& values) const;
	// Create child splits for a node or make terminal
	void split(dictionary& node, int max_depth, int min_size, int depth) const;
	// Build a decision tree
	dictionary build_tree(const std::vector<std::vector<T>>& train, int max_depth, int min_size) const;
	// Print a decision tree
	void print_tree(const dictionary& node, int depth = 0) const;
	// Make a prediction with a decision tree
	// predict() const;

private:
	std::vector<std::vector<T>> features;
	std::vector<T> labels;
	int samples_num = 0;
	int feature_length = 0;
	int classes_num = 0;
};

} // namespace ANN


#endif // FBC_NN_DECISION_TREE_HPP_
