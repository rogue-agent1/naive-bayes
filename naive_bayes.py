#!/usr/bin/env python3
"""naive_bayes - Gaussian and multinomial Naive Bayes classifier."""
import sys, json, math
from collections import Counter, defaultdict

class GaussianNB:
    def __init__(self):
        self.classes = []; self.priors = {}; self.means = {}; self.vars = {}
    
    def fit(self, X, y):
        self.classes = sorted(set(y))
        n = len(y)
        for c in self.classes:
            idx = [i for i in range(n) if y[i] == c]
            self.priors[c] = len(idx)/n
            feats = [[X[i][f] for i in idx] for f in range(len(X[0]))]
            self.means[c] = [sum(f)/len(f) for f in feats]
            self.vars[c] = [sum((x-m)**2 for x in f)/len(f)+1e-9 for f, m in zip(feats, self.means[c])]
    
    def _log_prob(self, x, c):
        lp = math.log(self.priors[c])
        for f in range(len(x)):
            lp += -0.5*math.log(2*math.pi*self.vars[c][f]) - (x[f]-self.means[c][f])**2/(2*self.vars[c][f])
        return lp
    
    def predict(self, X):
        return [max(self.classes, key=lambda c: self._log_prob(x, c)) for x in X]

class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha; self.classes = []; self.priors = {}; self.log_probs = {}
    
    def fit(self, X, y):
        self.classes = sorted(set(y)); n = len(y); nf = len(X[0])
        for c in self.classes:
            idx = [i for i in range(n) if y[i] == c]
            self.priors[c] = math.log(len(idx)/n)
            counts = [sum(X[i][f] for i in idx)+self.alpha for f in range(nf)]
            total = sum(counts)
            self.log_probs[c] = [math.log(c/total) for c in counts]
    
    def predict(self, X):
        results = []
        for x in X:
            scores = {c: self.priors[c]+sum(x[f]*self.log_probs[c][f] for f in range(len(x))) for c in self.classes}
            results.append(max(scores, key=scores.get))
        return results

def main():
    print("Naive Bayes demo\n")
    X = [[1.0,2.0],[1.5,1.8],[5.0,8.0],[6.0,9.0],[1.0,0.6],[2.0,1.0],[5.5,7.0],[7.0,8.5]]
    y = [0,0,1,1,0,0,1,1]
    gnb = GaussianNB(); gnb.fit(X, y)
    preds = gnb.predict([[3,4],[6,8],[1,1]])
    acc = sum(1 for p,t in zip(gnb.predict(X),y) if p==t)/len(y)
    print(f"Gaussian NB: accuracy={acc*100:.0f}%, test_preds={preds}")
    # Multinomial (text-like)
    X2 = [[3,0,1],[2,1,0],[0,3,2],[1,2,3],[4,0,0],[0,1,4]]
    y2 = [0,0,1,1,0,1]
    mnb = MultinomialNB(); mnb.fit(X2, y2)
    preds2 = mnb.predict([[2,0,1],[0,2,3]])
    print(f"Multinomial NB: test_preds={preds2}")

if __name__ == "__main__":
    main()
