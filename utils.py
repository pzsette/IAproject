import random
import csv

# Source: https://github.com/aimacode/aima-python/blob/master/learning.py

def removeall(item, seq):
    """Return a copy of seq (or string) with all occurrences of item removed."""
    if isinstance(seq, str):
        return seq.replace(item, '')
    else:
        return [x for x in seq if x != item]

def normalize(dist):
    """Multiply each number by a constant such that the sum is 1.0"""
    if isinstance(dist, dict):
        total = sum(dist.values())
        for key in dist:
            dist[key] = dist[key] / total
            assert 0 <= dist[key] <= 1, "Probabilities must be between 0 and 1."
        return dist
    total = sum(dist)
    return [(n / total) for n in dist]

def shuffled(iterable):
    """Randomly shuffle a copy of iterable."""
    items = list(iterable)
    random.shuffle(items)
    return items

def parse_file(path):
    """ Read a csv/txt file and return a list containing each row as a list of elements. """
    with open(path, 'r') as file:
        data = list(csv.reader(file))
    return data

def argmax_random_tie(seq, key=lambda x: x):
    """Return an element with highest fn(seq[i]) score; break ties at random."""
    return max(shuffled(seq), key=key)

def unique(seq):
    """Remove duplicate elements from seq. Assumes hashable elements."""
    return list(set(seq))
