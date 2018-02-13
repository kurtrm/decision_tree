"""
Module to test the decision tree class.
"""
import pytest
import random

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


table = []
for x in range(150):
    randy = random.randint(1, 101)
    table.append((range(randy), 1 - 1/randy))
table.append(([], 0))

gini_table = []
for i in range(100):
    new_randy = random.randint(1, 20)
    gini_table.append((range(new_randy), 1 - sum((1 / new_randy)**2 for _ in range(new_randy))))
gini_table.append(([25, 25, 75], 1 - sum((p / 125)**2 for p in [25, 25, 75])))


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


@pytest.mark.parametrize('val, expected', table)
def test_max_gini(val, expected, decision_tree):
    """
    Test that max_gini gives the correct number in the sequence.
    """
    assert decision_tree._max_gini(val) == expected


@pytest.mark.parametrize('val, expected', gini_table)
def test_gini(val, expected, decision_tree):
    """
    Test that gini returns the correct val.
    """
    assert decision_tree._gini(val) == expected
