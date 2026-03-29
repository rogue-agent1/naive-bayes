#!/usr/bin/env python3
"""naive_bayes: Gaussian Naive Bayes classifier."""
import math, sys
from collections import defaultdict

class GaussianNB:
    def __init__(self):
        self.classes = {}
        self.priors = {}

    def fit(self, X, y):
        by_class = defaultdict(list)
        for xi, yi in zip(X, y):
            by_class[yi].append(xi)
        total = len(y)
        for c, samples in by_class.items():
            self.priors[c] = len(samples) / total
            n_features = len(samples[0])
            stats = []
            for f in range(n_features):
                vals = [s[f] for s in samples]
                mean = sum(vals) / len(vals)
                var = sum((v - mean)**2 for v in vals) / len(vals) + 1e-9
                stats.append((mean, var))
            self.classes[c] = stats

    def _log_prob(self, x, c):
        log_p = math.log(self.priors[c])
        for xi, (mean, var) in zip(x, self.classes[c]):
            log_p += -0.5 * math.log(2 * math.pi * var) - (xi - mean)**2 / (2 * var)
        return log_p

    def predict(self, x):
        return max(self.classes.keys(), key=lambda c: self._log_prob(x, c))

    def predict_batch(self, X):
        return [self.predict(x) for x in X]

def test():
    X = [[1,1],[1,2],[2,1],[2,2],[5,5],[5,6],[6,5],[6,6]]
    y = [0,0,0,0,1,1,1,1]
    nb = GaussianNB()
    nb.fit(X, y)
    assert nb.predict([1.5, 1.5]) == 0
    assert nb.predict([5.5, 5.5]) == 1
    preds = nb.predict_batch([[0,0],[10,10]])
    assert preds == [0, 1]
    # Priors
    assert abs(nb.priors[0] - 0.5) < 0.01
    assert abs(nb.priors[1] - 0.5) < 0.01
    print("All tests passed!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test": test()
    else: print("Usage: naive_bayes.py test")
