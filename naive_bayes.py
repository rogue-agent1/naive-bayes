#!/usr/bin/env python3
"""Naive Bayes text classifier."""
import sys, math, re
from collections import defaultdict

class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_counts = defaultdict(int)
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.vocab = set()
    def tokenize(self, text):
        return re.findall(r"[a-z]+", text.lower())
    def train(self, texts, labels):
        for text, label in zip(texts, labels):
            self.class_counts[label] += 1
            for word in self.tokenize(text):
                self.word_counts[label][word] += 1
                self.vocab.add(word)
    def predict(self, text):
        tokens = self.tokenize(text)
        total = sum(self.class_counts.values())
        best_class, best_score = None, float('-inf')
        for cls in self.class_counts:
            score = math.log(self.class_counts[cls] / total)
            total_words = sum(self.word_counts[cls].values())
            for word in tokens:
                count = self.word_counts[cls].get(word, 0)
                score += math.log((count + self.alpha) / (total_words + self.alpha * len(self.vocab)))
            if score > best_score:
                best_score, best_class = score, cls
        return best_class
    def predict_proba(self, text):
        tokens = self.tokenize(text)
        total = sum(self.class_counts.values())
        scores = {}
        for cls in self.class_counts:
            score = math.log(self.class_counts[cls] / total)
            tw = sum(self.word_counts[cls].values())
            for word in tokens:
                count = self.word_counts[cls].get(word, 0)
                score += math.log((count + self.alpha) / (tw + self.alpha * len(self.vocab)))
            scores[cls] = score
        max_s = max(scores.values())
        exp_scores = {c: math.exp(s - max_s) for c, s in scores.items()}
        total_exp = sum(exp_scores.values())
        return {c: v / total_exp for c, v in exp_scores.items()}

def test():
    nb = NaiveBayes()
    texts = ["great movie loved it", "terrible awful waste", "fantastic brilliant amazing",
             "horrible boring bad", "wonderful excellent perfect", "worst disaster ugly"]
    labels = ["pos", "neg", "pos", "neg", "pos", "neg"]
    nb.train(texts, labels)
    assert nb.predict("great brilliant film") == "pos"
    assert nb.predict("terrible horrible movie") == "neg"
    proba = nb.predict_proba("great film")
    assert proba["pos"] > proba["neg"]
    print("  naive_bayes: ALL TESTS PASSED")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test": test()
    else: print("Naive Bayes classifier")
