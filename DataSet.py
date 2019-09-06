from utils import *
from collections import *
import copy

# Source: https://github.com/aimacode/aima-python/blob/master/learning.py

class DataSet:
    """A data set for a machine learning problem. It has the following fields:
    d.examples   A list of examples. Each one is a list of attribute values.

    d.target     The attribute that a learning algorithm will try to predict.
                 By default the final attribute.
    d.inputs     The list of attrs without the target.
    d.values     A list of lists: each sublist is the set of possible
                 values for the corresponding attribute. If initially None,
                 it is computed from the known examples by self.setproblem.
                 If not None, an erroneous value raises ValueError.
    d.name       Name of the data set (for output display only).
    """

    def __init__(self, examples=None, target=-1, name='', values=None):
        """Accepts any of DataSet's fields. Examples can also be a
        string or file from which to parse examples using parse_csv.
        Optional parameter: exclude, as documented in .setproblem().
        <DataSet(): 1 examples, 3 attributes>
        """
        self.name = name
        self.examples = examples
        self.target = target
        if values!=None:
            self.values=values
        else:
            self.values = list(map(unique, zip(*self.examples)))
        self.attrs = list(range(len(self.examples[0])))
        self.inputs = [a for a in self.attrs if a != self.target]

    def attrnum(self, attr):
        """Returns the number used for attr, which can be a name, or -n .. n-1."""
        if isinstance(attr, str):
            return self.attrnames.index(attr)
        elif attr < 0:
            return len(self.attrs) + attr
        else:
            return attr

    def sanitize(self, example):
        """Return a copy of example, with non-input attributes replaced by None."""
        return [attr_i if i in self.inputs else None
                for i, attr_i in enumerate(example)]

# Remove value from a dataset with a certain probability

def removeRadomValues(dataset, probability):
        examplesCopy = copy.deepcopy(dataset.examples)
        for example in examplesCopy:
            for x in range(len(example)):
                rndNumber = random.uniform(0, 1)
                if (rndNumber < probability and x != dataset.target):
                    example[x] = None
        return DataSet(name=dataset.name, examples=examplesCopy, target=dataset.target)

# Technique for replacing missing values as described in Mitchell (1997) 3.7.4

def countValueForAttribute(dataset):
    n = len(dataset.values)
    allValuesList = []
    for i in range(n):
        allValuesList.append([])
    for example in dataset.examples:
        for i in range(len(example)):
            allValuesList[i].append(example[i])

    valuesCounter = []
    for x in allValuesList:
        countedValues = Counter(x)
        del countedValues[None]
        valuesCounter.append(countedValues.most_common(1)[0][0])
    return valuesCounter

def replaceMissingValue(dataset):
    mostFrequentValueList = countValueForAttribute(dataset)
    for e in dataset.examples:
        for i in range(len(e)):
            if e[i] == None:
                e[i] = mostFrequentValueList[i]
