#!/usr/bin/env python3
"""Naive Bayes classifier — zero-dep implementation."""
import sys, math
from collections import defaultdict

class NaiveBayes:
    def __init__(self, smoothing=1.0):
        self.smoothing=smoothing
        self.class_counts=defaultdict(int)
        self.feature_counts=defaultdict(lambda:defaultdict(lambda:defaultdict(int)))
        self.vocab=defaultdict(set)
        self.total=0

    def fit(self, X, y):
        for xi,yi in zip(X,y):
            self.class_counts[yi]+=1; self.total+=1
            for j,v in enumerate(xi):
                self.feature_counts[yi][j][v]+=1
                self.vocab[j].add(v)

    def predict(self, X):
        return [self._predict_one(x) for x in X]

    def _predict_one(self, x):
        best_c=None; best_p=float('-inf')
        for c in self.class_counts:
            lp=math.log(self.class_counts[c]/self.total)
            for j,v in enumerate(x):
                n=self.feature_counts[c][j][v]+self.smoothing
                d=self.class_counts[c]+self.smoothing*len(self.vocab[j])
                lp+=math.log(n/d)
            if lp>best_p: best_p=lp; best_c=c
        return best_c

if __name__=="__main__":
    X=[["sunny","hot"],["sunny","hot"],["overcast","hot"],["rain","mild"],["rain","cool"],
       ["rain","cool"],["overcast","cool"],["sunny","mild"],["sunny","cool"],["rain","mild"]]
    y=["no","no","yes","yes","yes","no","yes","no","yes","yes"]
    nb=NaiveBayes(); nb.fit(X,y)
    tests=[["sunny","cool"],["overcast","hot"],["rain","mild"]]
    for t in tests: print(f"{t} -> {nb._predict_one(t)}")
    preds=nb.predict(X); acc=sum(p==a for p,a in zip(preds,y))/len(y)
    print(f"Training accuracy: {acc:.0%}")
