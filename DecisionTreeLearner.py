import math
from utils import *
from DecisionLeaf import *
from DecisionFork import *

# Source: https://github.com/aimacode/aima-python/blob/master/learning.py

def DecisionTreeLearner(dataset):

    target = dataset.target
    values = dataset.values

    def decision_tree_learning(examples, attrs, parent_examples=()):

        if len(examples) == 0:
            return plurality_value(parent_examples)
        elif all_same_class(examples):
            return DecisionLeaf(examples[0][target])
        elif len(attrs) == 0:
            return plurality_value(examples)
        else:
            A = choose_attribute(attrs, examples)
            tree = DecisionFork(A, plurality_value(examples))
            for (v_k, exs) in split_by(A, examples):
                subtree = decision_tree_learning(exs, removeall(A, attrs), examples)
                tree.add(v_k, subtree)
            return tree

    def count(attr, val, examples):
        """Count the number of examples that have example[attr] = val."""
        return sum(e[attr] == val for e in examples)

    def plurality_value(examples):
        """Return the most popular target value for this set of examples.
        (If target is binary, this is the majority; otherwise plurality.)"""
        popular = argmax_random_tie(values[target], key=lambda v: count(target, v, examples))
        return DecisionLeaf(popular)

    def all_same_class(examples):
        """Are all these examples in the same target class?"""
        class0 = examples[0][target]
        return all(e[target] == class0 for e in examples)

    def choose_attribute(attrs, examples):
        """Choose the attribute with the highest information gain."""
        return argmax_random_tie(attrs, key=lambda a: information_gain(a, examples))

    def information_gain(attr, examples):
        """Return the expected reduction in entropy from splitting by attr."""
        def I(examples):
            return information_content([count(target, v, examples) for v in values[target]])
        N = len(examples)
        remainder = sum((len(examples_i)/N) * I(examples_i) for (v, examples_i) in split_by(attr, examples))
        return I(examples) - remainder

    def split_by(attr, examples):
        """Return a list of (val, examples) pairs for each val of attr."""
        return [(v, [e for e in examples if e[attr] == v]) for v in values[attr]]

    return decision_tree_learning(dataset.examples, dataset.inputs)


def information_content(values):
    """Number of bits to represent the probability distribution in values."""
    probabilities = normalize(removeall(0, values))
    return sum(-p * math.log2(p) for p in probabilities)