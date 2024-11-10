import math
from collections import Counter, defaultdict
from typing import List


def entropy(Plist):
    if len(Plist):
        result = 0
        for x in Plist:
            result += (-x) * math.log(x, 2)
        return result
    else:
        return 0

def norm(Olist):
    s = sum(Olist)
    return [o / s for o in Olist]

def count(Olist):
    x_dict = defaultdict(lambda: 0.0)
    for x in Olist:
        x_dict[x] += 1
    cc = [c for _,c in x_dict.items()]
    #print(cc)
    return cc

def item_entropy(answers: List) -> float:
    return entropy(norm(count(answers)))


def length_normalized_entropy(log_probs: List[float]) -> float:
    entropy = -sum(log_probs)
    return entropy / len(log_probs)
