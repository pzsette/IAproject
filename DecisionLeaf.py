# Source: https://github.com/aimacode/aima-python/blob/master/learning.py


class DecisionLeaf:
    """A leaf of a decision tree holds just a result."""

    def __init__(self, result):
        self.result = result

    def __call__(self, example):
        return self.result
