"""
Module containing the DecisionTree class.
"""
from collections import Counter
from math import log2


class Node:
    """
    Node that contains all information required in order to make predictions and pass
    information on for further evaluation to other nodes.
    """

    def __init__(self, threshold, samples_count, values, classification, feature, gini=None):
        """
        """
        self.threshold = threshold
        self.samples_count = samples_count
        self.values = values
        self.classification = classification
        self.feature = feature
        self.left = None
        self.right = None
        self.gini = gini


class DecisionTree:
    """
    A crude implentation of a decision tree that can either use the Gini index
    or entropy (information gain).
    """

    def __init__(self, max_depth=2):
        """
        """
        self.root = None
        self.max_depth = max_depth

    def train(self, labeled_data, method='gini'):
        """
        """
        if method == 'gini':
            self._cart(labeled_data)
        elif method == 'entropy':
            pass

    def predict(self, data):
        """
        Bsae
        """
        pass

    def _gini(self, labels):
        """
        Calculates the gini impurity for a set of labels.
        """
        total = len(labels)
        label_counts = Counter(labels).values()
        return 1 - sum((p / total)**2
                       for p in label_counts
                       if p)

    def _entropy(self, labels):
        """
        Calculates entropy for a set of labels.
        """
        total = len(labels)
        label_counts = Counter(labels).values()
        return -sum((p / total) * log2(p / total)
                    for p in label_counts
                    if p)

    def _max_gini(self, labels):
        """
        This function used to determine if CART should make a node a leaf node depending on
        how close the gini impurity is to the theoretical max. The max can be represented as follows:
        a_n = 1 - 1/x, x >= 1

        For a given set of labels, the gini impurity will approach the maximum.
        """
        num_labels = len(Counter(labels))
        return 1 - 1/num_labels if num_labels else 0

    def _id3():
        """
        Builds a decision tree using the ID3 algorithm.
        """
        pass

    def _cart(self, labeled_data, depth=0, gini_split_treshold=.25):
        """
        Classification and Regression Tree implementation.
        """
        # Starts with all the data and runs the cost function on each feature to get the best starting split.
        current_depth = depth
        labels = [label[1] for label in labeled_data]
        gini_threshold = gini_split_treshold * self._max_gini(labels)
        if depth >= self.max_depth or self._gini(labels) <= gini_threshold:
            counts = Counter(labels)
            max_val = max(counts.values())
            for key, value in counts.items():
                if counts[key] == max_val:
                    classification = key
            return Node(None, len(labeled_data), labeled_data, classification, None, self._gini(labels))

        lowest_cost = float('inf')
        for feature in labeled_data[0][0].keys():
            gini_calculations = self._gini_cost(labeled_data, feature)
            if gini_calculations[0] < lowest_cost:
                lowest_cost, threshold, left_samples, right_samples = gini_calculations
                chosen_feature = feature

        if self.root is None:
            self.root = Node(threshold, len(labeled_data), labeled_data, left_samples[0][1], chosen_feature, self._gini(labels))
            node = self.root
        else:
            node = Node(threshold, len(labeled_data), labeled_data, left_samples[0][1], chosen_feature, self._gini(labels))

        if current_depth < self.max_depth:
            current_depth += 1
            node.left = self._cart(left_samples, current_depth)
            node.right = self._cart(right_samples, current_depth)
        return node

    def _gini_cost(self, labeled_data, feature_name):
        """
        Calculate the cost function as part of the CART algorithm.

        For continuous data, multiplying by 100 then dividing the individual numbers by 100 was an operation
        chosen arbitrarily to find the best number to split the data on. It's a magic number for sure.
        """
        minim = []
        cost_min = float('inf')
        len_labeled_data = len(labeled_data)
        feature_data = [row[0][feature_name] for row in labeled_data]
        max_feature = int(max(feature_data)) * 100
        min_feature = int(min(feature_data)) * 100
        span = (i / 100 for i in range(min_feature, max_feature + 1))
        for x in span:
            left = []
            right = []
            for i, num in enumerate(feature_data):
                if num <= x:
                    left.append(i)
                else:
                    right.append(i)
            left_labels = [labeled_data[idx][1] for idx in left]
            right_labels = [labeled_data[idx][1] for idx in right]
            cost = len(left_labels) * self._gini(left_labels) / len_labeled_data + len(right_labels) * self._gini(right_labels) / len_labeled_data
            if cost < cost_min:
                cost_min = cost
                minim = []
                minim.append(x)
                left_samples = [labeled_data[idx] for idx in left]
                right_samples = [labeled_data[idx] for idx in right]
            elif cost == cost_min:
                minim.append(x)
        avg_minimums = sum(minim) / len(minim)
        return cost_min, avg_minimums, left_samples, right_samples
