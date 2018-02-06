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

    def __init__(self, threshold, samples, values, classification):
        """
        """
        self.threshold = threshold
        self.samples = samples
        self.values = values
        self.classification = classification
        self.gini = self._gini(values)
        self.left = None
        self.right = None






class DecisionTree:
    """
    A crude implentation of a decision tree that can either use the Gini index
    or entropy (information gain).
    """

    def __init__(self):
        """
        """
        pass

    def train(self, X, y, method='entropy'):
        """
        """
        pass

    def predict():
        """
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

    def _id3():
        """
        Builds a decision tree using the ID3 algorithm.
        """
        pass

    def _cart():
        """
        Classification and Regression Tree implementation.
        """

    def _gini_cost(self, labeled_data, feature_name):
        """
        Calculate the cost function as part of the CART algorithm.
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
                left_node = [labeled_data[idx] for idx in left]
                right_node = [labeled_data[idx] for idx in right]
            elif cost == cost_min:
                minim.append(x)
        avg_minimums = sum(minim) / len(minim)
        return avg_minimums, left_node, right_node
