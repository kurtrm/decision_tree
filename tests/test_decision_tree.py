"""
Module to test the decision tree class.
"""
import pytest


@pytest.fixture
def decision_tree():
    """
    Decision tree fixture.
    """
    from .src.decision_tree import DecisionTree
    from .iris_petal_data import iris_data
    tree = DecisionTree()
    tree.train(iris_data)
    return tree
