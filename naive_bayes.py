#!/usr/bin/env python3
"""naive_bayes - Naive Bayes text classifier."""
import sys,math,re
from collections import defaultdict,Counter
class NaiveBayes:
    def __init__(s):s.classes={};s.vocab=set();s.priors={}
    def train(s,docs):
        class_docs=defaultdict(list)
        for text,label in docs:class_docs[label].append(text)
        total=len(docs)
        for label,texts in class_docs.items():
            s.priors[label]=len(texts)/total;words=[]
            for t in texts:words.extend(re.findall(r"\w+",t.lower()))
            s.classes[label]=Counter(words);s.vocab.update(words)
    def predict(s,text):
        words=re.findall(r"\w+",text.lower());best_label=None;best_score=float("-inf")
        for label in s.classes:
            score=math.log(s.priors[label]);total=sum(s.classes[label].values())
            for w in words:score+=math.log((s.classes[label][w]+1)/(total+len(s.vocab)))
            if score>best_score:best_score=score;best_label=label
        return best_label,best_score
if __name__=="__main__":
    train_data=[("great movie loved it","positive"),("terrible waste of time","negative"),
        ("amazing performance brilliant","positive"),("awful boring bad","negative"),
        ("fantastic wonderful excellent","positive"),("horrible worst ever","negative")]
    nb=NaiveBayes();nb.train(train_data)
    tests=["great performance","terrible movie","wonderful time","bad and boring"]
    for t in tests:label,score=nb.predict(t);print(f"  '{t}' → {label} ({score:.2f})")
