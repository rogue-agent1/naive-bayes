#!/usr/bin/env python3
"""Naive Bayes — text classification."""
import math, sys, re
from collections import Counter, defaultdict

class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha; self.class_counts = Counter()
        self.word_counts = defaultdict(Counter); self.vocab = set()
    def _tokenize(self, text): return re.findall(r'\w+', text.lower())
    def fit(self, texts, labels):
        for text, label in zip(texts, labels):
            self.class_counts[label] += 1
            for word in self._tokenize(text):
                self.word_counts[label][word] += 1; self.vocab.add(word)
    def predict(self, text):
        tokens = self._tokenize(text); scores = {}
        total = sum(self.class_counts.values()); V = len(self.vocab)
        for cls, count in self.class_counts.items():
            score = math.log(count / total)
            total_words = sum(self.word_counts[cls].values())
            for word in tokens:
                wc = self.word_counts[cls].get(word, 0)
                score += math.log((wc + self.alpha) / (total_words + self.alpha * V))
            scores[cls] = score
        return max(scores, key=scores.get), scores

if __name__ == "__main__":
    texts = ["great movie loved it", "terrible waste of time", "amazing performance",
             "boring and dull", "fantastic story", "awful acting horrible plot",
             "wonderful cinematography", "worst film ever", "brilliant masterpiece", "so bad"]
    labels = ["pos","neg","pos","neg","pos","neg","pos","neg","pos","neg"]
    nb = NaiveBayes(); nb.fit(texts, labels)
    tests = ["great performance amazing", "terrible and boring", "loved the story"]
    for t in tests:
        pred, scores = nb.predict(t)
        print(f"  '{t}' -> {pred} (pos={scores.get('pos',0):.2f}, neg={scores.get('neg',0):.2f})")
