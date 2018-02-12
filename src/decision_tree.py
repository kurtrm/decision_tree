"""
Module containing the DecisionTree class.
"""
from collections import Counter
from math import log2


class Node:
    """
    Node that contains all information required in order to make predictions
    and pass information on for further evaluation to other nodes.
    """

    def __init__(self, samples_count, values, classification,
                 threshold=None,
                 feature=None,
                 gini=None):
        """
        Initialize a node.
        """
        self.samples_count = samples_count
        self.values = values
        self.classification = classification
        self.threshold = threshold
        self.feature = feature
        self.gini = gini
        self.left = None
        self.right = None

    def __repr__(self):
        """
        Return a simple representation of a node object.
        """
        return '<[Node] gini={:.3f} feature={}>'.format(self.gini,
                                                        self.feature)

    def __str__(self):
        """
        Return string representation of a node. Offers more info than
        repr.
        """
        values = list(Counter([value[1] for value in self.values]).values())
        return """
    {} <= {:.3f}
    gini = {:.3f}
    samples = {}
    values = {}
    class = {}""".format(self.feature,
                         self.threshold,
                         self.gini,
                         self.samples_count,
                         values,
                         self.classification)


class DecisionTree:
    """
    A crude implementation of a decision tree that, for the time being,
    uses the gini index to create a binary decision tree.
    """

    def __init__(self, max_depth=2):
        """
        Instantiate a decision tree with a default depth of 2.
        """
        if max_depth <= 0 or not isinstance(max_depth, int):
            raise ValueError('max_depth must be '
                             'an integer greater than zero')
        self.max_depth = max_depth
        self.root = None

    def train(self, labeled_data, method='gini', gini_split_threshold=.25):
        """
        labeled_data is an iterable containing iterables of a dictionary and
        a corresponding label. For example:
        data = [
            ({'feature_1': 3, 'feature_2': 4}, 'label_2'),
            ({'feature_1': 2, 'feature_2': 3}, 'label_4')
        ]

        Note: Only works with the gini method right now.

        The gini_split_threshold default is arbitrary.
        """
        if method not in ['gini', 'entropy']:
            raise ValueError("method parameter must be "
                             "either 'gini' or 'entropy'")

        if gini_split_threshold < 0 or gini_split_threshold > 1:
            raise ValueError('gini_split_threshold argument must '
                             'be a float between 0 and 1')

        if method == 'gini':
            self._cart(labeled_data, gini_split_threshold=gini_split_threshold)
        elif method == 'entropy':
            raise NotImplementedError('This feature is under construction')

    def predict(self, data):
        """
        This will take a list of dictionaries and return a list of
        predicted labels.
        """
        predictions = []
        start_node = self.root
        for piece in data:
            current_node = start_node
            while current_node.feature:
                    if piece[current_node.feature] <= current_node.threshold:
                        current_node = current_node.left
                    else:
                        current_node = current_node.right
            predictions.append(current_node.classification)

        return predictions

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
        This function used to determine if CART should make a node a leaf node
        depending on how close the gini impurity is to the theoretical max.
        The max can be represented as follows:
        a_n = 1 - 1/x, x >= 1

        For a given set of labels, the gini impurity will approach the maximum
        the more spread out the data.
        """
        num_labels = len(Counter(labels))
        return 1 - 1/num_labels if num_labels else 0

    def _id3():
        """
        Builds a decision tree using the ID3 algorithm.
        """
        pass

    def _cart(self, labeled_data, gini_split_threshold, depth=0):
        """
        Classification and Regression Tree (CART) implementation.

        This version of the algorithm only works with continuous data. It will
        search for the split in the data that produces the smallest gini value
        of the available features. For a the number of unique features, there
        exists a maximum of the gini. We use this value to determine what the
        maximum allowable gini valuable to make a node a leaf node. For
        example, if we pass 0 as the threshold, it will continue to split
        the data recursively until it hits the max_depth or the gini hits
        zero. That's not very practical.
        """
        labels = [label[1] for label in labeled_data]
        gini_threshold = gini_split_threshold * self._max_gini(labels)
        if depth >= self.max_depth or self._gini(labels) <= gini_threshold:
            counts = Counter(labels)
            max_val = max(counts.values())
            for key, value in counts.items():
                if value == max_val:
                    classification = key
            return Node(len(labeled_data), labeled_data, classification, gini=self._gini(labels))

        lowest_cost = float('inf')
        for feature in labeled_data[0][0].keys():
            gini_calculations = self._gini_cost(labeled_data, feature)
            if gini_calculations[0] < lowest_cost:
                lowest_cost, threshold, left_samples, right_samples = gini_calculations
                chosen_feature = feature

        if self.root is None:
            self.root = Node(len(labeled_data), labeled_data, left_samples[0][1], threshold, chosen_feature, self._gini(labels))
            node = self.root
        else:
            node = Node(len(labeled_data), labeled_data, left_samples[0][1], threshold, chosen_feature, self._gini(labels))

        if depth < self.max_depth:
            depth += 1
            node.left = self._cart(left_samples, gini_split_threshold, depth)
            node.right = self._cart(right_samples, gini_split_threshold, depth)

        return node

    def _gini_cost(self, labeled_data, feature_name):
        """
        Calculate the cost function as part of the CART algorithm.

        For continuous data, multiplying by 100 then dividing the individual
        numbers by 100 was an operation chosen arbitrarily to find the best
        number to split the data on. It's a magic number for sure.
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
