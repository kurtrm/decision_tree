"""
Module containing the DecisionTree class.
"""
from collections import Counter


class DecisionTree:
    """
    A crude implentation of a decision tree that can either use the Gini index
    or entropy (information gain).
    """

    def __init__(self):
        """
        """
        pass

    def __call__():
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
        """
        pass

    def _id3():
        """
        """
        pass

    def _cart():
        """
        """
        pass

