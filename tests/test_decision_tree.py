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


predictions = [
    {'petal length (cm)': 1.4, 'petal width (cm)': 0.2},
    {'petal length (cm)': 1.4, 'petal width (cm)': 0.2},
    {'petal length (cm)': 1.3, 'petal width (cm)': 0.2},
    {'petal length (cm)': 1.5, 'petal width (cm)': 0.2},
    {'petal length (cm)': 1.4, 'petal width (cm)': 0.2},
    {'petal length (cm)': 5.5, 'petal width (cm)': 1.8},
    {'petal length (cm)': 4.5, 'petal width (cm)': 1.6},
]


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


def test_gini_cost(decision_tree):
    """
    Test the gini_cost method on the available iris data.
    """
    from tests.iris_petal_data import iris_data
    gini, avg_costs, _, _ = decision_tree._gini_cost(iris_data,
                                                     'petal length (cm)')
    assert (gini, avg_costs) == (pytest.approx(1/3, .1),
                                 pytest.approx(2.45, .1))


def test_gini_cost_child_right(decision_tree):
    """
    Test hte gini_cost on a child of the root node.
    """
    from tests.iris_petal_data import iris_data
    _, _, left, right = decision_tree._gini_cost(iris_data,
                                                 'petal length (cm)')
    gini, avg_costs, _, _ = decision_tree._gini_cost(right,
                                                     'petal width (cm)')
    assert (gini, avg_costs) == (pytest.approx(.11, .1),
                                 pytest.approx(1.75, .1))


def test_train_errors(decision_tree):
    """
    Performing a unit test on _cart is needless since
    the train() method will call it.
    """
    from tests.iris_petal_data import iris_data
    for error in [('chaos', .3), ('entropy', -.25), ('gini', 1.25)]:
        with pytest.raises(ValueError):
            decision_tree.train(iris_data, *error)


def test_train_establishes_root(decision_tree):
    """
    Ensure we get the correct nodes back when training on the
    iris petal data set.
    """
    from tests.iris_petal_data import iris_data
    from src.decision_tree import Node
    decision_tree.train(iris_data)
    assert isinstance(decision_tree.root, Node)


def test_expected_tree_shape(decision_tree):
    """
    Ensure nodes have None at left and right values as expected.
    """
    from tests.iris_petal_data import iris_data
    decision_tree.train(iris_data)
    assert not all([decision_tree.root.left.left,
                   decision_tree.root.left.right,
                   decision_tree.root.right.right.right,
                   decision_tree.root.right.right.left,
                   decision_tree.root.right.left.right,
                   decision_tree.root.right.left.left])


@pytest.fixture
def loaded_tree():
    """
    Loaded tree following testing.
    """
    from src.decision_tree import DecisionTree
    from tests.iris_petal_data import iris_data
    tree = DecisionTree(max_depth=2)
    tree.train(iris_data)
    return tree


def test_root_node_attributes(loaded_tree):
    """
    Look at root node attributes.
    """
    root = loaded_tree.root
    assert all([
            root.samples_count == 150,
            root.values,
            root.classification == 'setosa',
            root.threshold == pytest.approx(2.45, .1),
            root.feature == 'petal length (cm)',
            root.gini == pytest.approx(2/3, .1),
            root.left,
            root.right
        ])


def test_left_leaf_attributes(loaded_tree):
    """
    Look at the left most leaf node attributes.
    """
    leaf = loaded_tree.root.left
    assert all([
            leaf.samples_count == 50,
            leaf.values,
            leaf.classification == 'setosa',
            not leaf.threshold,
            not leaf.feature,
            leaf.gini == 0,
            not leaf.left,
            not leaf.right
        ])


def test_right_leaf_attributes(loaded_tree):
    """
    Look at the right child of the root node.
    """
    leaf = loaded_tree.root.right
    assert all([
            leaf.samples_count == 100,
            leaf.values,
            leaf.classification == 'versicolor',
            leaf.threshold == pytest.approx(1.75, .1),
            leaf.feature == 'petal width (cm)',
            leaf.gini == .5,
            leaf.left,
            leaf.right
        ])


def test_right_children_left(loaded_tree):
    """
    Look at the children of the right node.
    """
    leaf = loaded_tree.root.right.left
    assert all([
            leaf.samples_count == 54,
            leaf.values,
            leaf.classification == 'versicolor',
            not leaf.threshold,
            not leaf.feature,
            leaf.gini == pytest.approx(0.168, .1),
            not leaf.left,
            not leaf.right
        ])


def test_right_children_right(loaded_tree):
    """
    Look at the children of the right node.
    """
    leaf = loaded_tree.root.right.right
    assert all([
            leaf.samples_count == 46,
            leaf.values,
            leaf.classification == 'virginica',
            not leaf.threshold,
            not leaf.feature,
            leaf.gini == pytest.approx(0.043, .1),
            not leaf.left,
            not leaf.right
        ])


def test_predictions(loaded_tree):
    """
    Ensure we get expected predictions.
    """
    assert loaded_tree.predict(predictions) == ['setosa',
                                                'setosa',
                                                'setosa',
                                                'setosa',
                                                'setosa',
                                                'virginica',
                                                'versicolor']


def test_id3(decision_tree):
    """
    For now, assert that we get None when trying to train a tree using
    the id3 algorithm.
    """
    from src.decision_tree import DecisionTree
    from tests.iris_petal_data import iris_data
    tree = DecisionTree()
    tree.train(iris_data, method='entropy')
    assert tree.root is None
