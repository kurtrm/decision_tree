"""
Module to test the decision tree class.
"""
import pytest


@pytest.fixture
def decision_tree():
    """
    Decision tree fixture.
    """
    from src.decision_tree import DecisionTree
    from tests.iris_petal_data import iris_data
    tree = DecisionTree()
    tree.train(iris_data)
    return tree


def test_instantiation():
    """
    Test instantiation. I know it's cheap, but it's a start.
    """
    from src.decision_tree import DecisionTree
    tree = DecisionTree()
    assert isinstance(tree, DecisionTree)


def test_instantiation_with_depth():
    """
    Test that we hit the ValueError with an inappropriate max_depth
    input.
    """
    from src.decision_tree import DecisionTree
    with pytest.raises(ValueError):
        for val in [-1, 0, .5]:
            tree = DecisionTree(max_depth=val)

