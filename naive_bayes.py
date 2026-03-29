#!/usr/bin/env python3
"""Naive Bayes text classifier from scratch."""
import sys, math, re
from collections import Counter, defaultdict

class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha; self.class_counts = Counter()
        self.word_counts = defaultdict(Counter); self.vocab = set()

    def tokenize(self, text):
        return re.findall(r"\b[a-z]+\b", text.lower())

    def train(self, texts, labels):
        for text, label in zip(texts, labels):
            self.class_counts[label] += 1
            for word in self.tokenize(text):
                self.word_counts[label][word] += 1
                self.vocab.add(word)

    def predict(self, text):
        words = self.tokenize(text); total = sum(self.class_counts.values())
        best_label, best_score = None, -float("inf")
        for label in self.class_counts:
            score = math.log(self.class_counts[label] / total)
            total_words = sum(self.word_counts[label].values())
            for word in words:
                count = self.word_counts[label].get(word, 0)
                score += math.log((count + self.alpha) / (total_words + self.alpha * len(self.vocab)))
            if score > best_score: best_score = score; best_label = label
        return best_label, best_score

    def predict_proba(self, text):
        words = self.tokenize(text); total = sum(self.class_counts.values())
        scores = {}
        for label in self.class_counts:
            score = math.log(self.class_counts[label] / total)
            total_words = sum(self.word_counts[label].values())
            for word in words:
                count = self.word_counts[label].get(word, 0)
                score += math.log((count + self.alpha) / (total_words + self.alpha * len(self.vocab)))
            scores[label] = score
        max_s = max(scores.values())
        exp_scores = {k: math.exp(v - max_s) for k, v in scores.items()}
        total_exp = sum(exp_scores.values())
        return {k: v / total_exp for k, v in exp_scores.items()}

def demo():
    print("=== Naive Bayes Sentiment Demo ===")
    texts = ["great movie loved it", "terrible awful waste", "amazing fantastic film",
             "horrible boring bad", "excellent wonderful", "worst movie ever",
             "brilliant masterpiece", "disappointing dull", "fun exciting thrilling", "ugly stupid"]
    labels = ["pos","neg","pos","neg","pos","neg","pos","neg","pos","neg"]
    nb = NaiveBayes(); nb.train(texts, labels)
    tests = ["great film", "terrible waste", "amazing but boring", "loved the movie"]
    for t in tests:
        label, _ = nb.predict(t)
        probs = nb.predict_proba(t)
        print(f"  \"{t}\" -> {label} ({', '.join(f'{k}:{v:.2%}' for k,v in sorted(probs.items()))})")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        nb = NaiveBayes(); texts, labels = [], []
        with open(sys.argv[2]) as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2: labels.append(parts[0]); texts.append(parts[1])
        nb.train(texts, labels)
        print(f"Trained on {len(texts)} examples, {len(nb.class_counts)} classes")
        if len(sys.argv) > 3:
            label, _ = nb.predict(sys.argv[3])
            print(f"Prediction: {label}")
    else: demo()

if __name__ == "__main__": main()
