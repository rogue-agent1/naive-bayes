#!/usr/bin/env python3
"""naive_bayes - Gaussian and Multinomial Naive Bayes classifiers."""
import sys, math
from collections import Counter, defaultdict

class GaussianNB:
    def __init__(self):
        self.classes = []
        self.means = {}
        self.vars = {}
        self.priors = {}

    def fit(self, X, y):
        self.classes = list(set(y))
        n = len(y)
        for c in self.classes:
            Xc = [X[i] for i in range(n) if y[i] == c]
            nc = len(Xc)
            self.priors[c] = nc / n
            ndim = len(Xc[0])
            self.means[c] = [sum(x[j] for x in Xc) / nc for j in range(ndim)]
            self.vars[c] = [sum((x[j] - self.means[c][j])**2 for x in Xc) / nc + 1e-9 for j in range(ndim)]

    def _log_prob(self, x, c):
        lp = math.log(self.priors[c])
        for j in range(len(x)):
            lp += -0.5 * math.log(2 * math.pi * self.vars[c][j])
            lp += -0.5 * (x[j] - self.means[c][j])**2 / self.vars[c][j]
        return lp

    def predict(self, x):
        return max(self.classes, key=lambda c: self._log_prob(x, c))

class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_log_prior = {}
        self.feature_log_prob = {}
        self.classes = []

    def fit(self, X, y):
        self.classes = list(set(y))
        n = len(y)
        ndim = len(X[0])
        for c in self.classes:
            Xc = [X[i] for i in range(n) if y[i] == c]
            self.class_log_prior[c] = math.log(len(Xc) / n)
            totals = [sum(x[j] for x in Xc) + self.alpha for j in range(ndim)]
            total_sum = sum(totals)
            self.feature_log_prob[c] = [math.log(t / total_sum) for t in totals]

    def predict(self, x):
        def score(c):
            return self.class_log_prior[c] + sum(x[j] * self.feature_log_prob[c][j] for j in range(len(x)))
        return max(self.classes, key=score)

def test():
    X = [[1,1],[2,2],[3,3],[8,8],[9,9],[10,10]]
    y = [0,0,0,1,1,1]
    gnb = GaussianNB()
    gnb.fit(X, y)
    assert gnb.predict([2, 2]) == 0
    assert gnb.predict([9, 9]) == 1
    assert abs(gnb.priors[0] - 0.5) < 0.01
    X2 = [[3,0,1],[2,1,0],[0,1,3],[1,0,2]]
    y2 = ["pos","pos","neg","neg"]
    mnb = MultinomialNB()
    mnb.fit(X2, y2)
    assert mnb.predict([3,0,0]) == "pos"
    assert mnb.predict([0,0,3]) == "neg"
    print("All tests passed!")

if __name__ == "__main__":
    test() if "--test" in sys.argv else print("naive_bayes: Naive Bayes classifiers. Use --test")
