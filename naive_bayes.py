#!/usr/bin/env python3
"""Naive Bayes text classifier from scratch."""
import sys, math, re, json
from collections import Counter, defaultdict

class NaiveBayes:
    def __init__(self):
        self.class_counts = Counter()
        self.word_counts = defaultdict(Counter)
        self.vocab = set()
    def train(self, texts, labels):
        for text, label in zip(texts, labels):
            self.class_counts[label] += 1
            words = re.findall(r'\w+', text.lower())
            for w in words:
                self.word_counts[label][w] += 1
                self.vocab.add(w)
    def predict(self, text):
        words = re.findall(r'\w+', text.lower())
        total = sum(self.class_counts.values())
        scores = {}
        for label, count in self.class_counts.items():
            score = math.log(count / total)
            total_words = sum(self.word_counts[label].values())
            for w in words:
                score += math.log((self.word_counts[label][w] + 1) / (total_words + len(self.vocab)))
            scores[label] = score
        return max(scores, key=scores.get), scores

if __name__ == '__main__':
    if '--demo' in sys.argv:
        texts = ["great movie loved it","terrible waste of time","amazing performance",
                 "awful horrible bad","wonderful beautiful film","boring stupid movie",
                 "excellent masterpiece","worst film ever made","fantastic acting",
                 "dull and lifeless"]
        labels = ["pos","neg","pos","neg","pos","neg","pos","neg","pos","neg"]
        nb = NaiveBayes()
        nb.train(texts, labels)
        tests = ["great film", "terrible acting", "wonderful movie", "awful waste"]
        for t in tests:
            pred, scores = nb.predict(t)
            print(f"  '{t}' → {pred} (scores: {', '.join(f'{k}={v:.2f}' for k,v in scores.items())})")
    else:
        print("Usage: naive_bayes.py --demo")
