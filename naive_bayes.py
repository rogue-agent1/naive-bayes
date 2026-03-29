#!/usr/bin/env python3
"""Naive Bayes classifier. Zero dependencies."""
import math
from collections import Counter, defaultdict

class GaussianNB:
    def __init__(self):
        self.classes = []; self.means = {}; self.vars = {}; self.priors = {}

    def fit(self, X, y):
        self.classes = list(set(y))
        n = len(y)
        for c in self.classes:
            Xc = [X[i] for i in range(n) if y[i] == c]
            self.priors[c] = len(Xc) / n
            d = len(X[0])
            self.means[c] = [sum(x[j] for x in Xc)/len(Xc) for j in range(d)]
            self.vars[c] = [sum((x[j]-self.means[c][j])**2 for x in Xc)/len(Xc)+1e-9 for j in range(d)]
        return self

    def _log_prob(self, x, c):
        lp = math.log(self.priors[c])
        for j in range(len(x)):
            lp -= 0.5*math.log(2*math.pi*self.vars[c][j])
            lp -= (x[j]-self.means[c][j])**2 / (2*self.vars[c][j])
        return lp

    def predict_one(self, x):
        return max(self.classes, key=lambda c: self._log_prob(x, c))

    def predict(self, X):
        return [self.predict_one(x) for x in X]

class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha; self.log_priors = {}; self.log_likelihoods = {}

    def fit(self, X, y):
        classes = set(y); n = len(y); vocab_size = len(X[0])
        for c in classes:
            Xc = [X[i] for i in range(n) if y[i] == c]
            self.log_priors[c] = math.log(len(Xc) / n)
            totals = [sum(x[j] for x in Xc) + self.alpha for j in range(vocab_size)]
            total_sum = sum(totals)
            self.log_likelihoods[c] = [math.log(t / total_sum) for t in totals]
        return self

    def predict_one(self, x):
        scores = {}
        for c in self.log_priors:
            scores[c] = self.log_priors[c] + sum(x[j]*self.log_likelihoods[c][j] for j in range(len(x)))
        return max(scores, key=scores.get)

    def predict(self, X): return [self.predict_one(x) for x in X]

if __name__ == "__main__":
    X = [[1,1],[2,2],[3,3],[8,8],[9,9],[10,10]]
    y = [0,0,0,1,1,1]
    nb = GaussianNB().fit(X, y)
    print(f"Predict [5,5]: {nb.predict_one([5,5])}")
