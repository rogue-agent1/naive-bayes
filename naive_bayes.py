#!/usr/bin/env python3
"""naive_bayes - Text classifier."""
import sys,argparse,json,math,re
from collections import Counter,defaultdict
class NaiveBayes:
    def __init__(self):self.class_counts=Counter();self.word_counts=defaultdict(Counter);self.vocab=set()
    def train(self,texts,labels):
        for text,label in zip(texts,labels):
            self.class_counts[label]+=1
            words=re.findall(r"\w+",text.lower())
            for w in words:self.word_counts[label][w]+=1;self.vocab.add(w)
    def predict(self,text):
        words=re.findall(r"\w+",text.lower())
        total=sum(self.class_counts.values());scores={}
        for label in self.class_counts:
            score=math.log(self.class_counts[label]/total)
            total_words=sum(self.word_counts[label].values())
            for w in words:
                score+=math.log((self.word_counts[label][w]+1)/(total_words+len(self.vocab)))
            scores[label]=round(score,4)
        best=max(scores,key=scores.get)
        return best,scores
def main():
    p=argparse.ArgumentParser(description="Naive Bayes")
    p.add_argument("--demo",action="store_true")
    p.add_argument("--predict",help="Text to classify")
    args=p.parse_args()
    nb=NaiveBayes()
    texts=["great movie loved it","terrible film waste of time","amazing performance brilliant","awful boring disaster","excellent wonderful masterpiece","horrible ugly terrible mess","fun exciting thrilling","dull tedious painful"]
    labels=["pos","neg","pos","neg","pos","neg","pos","neg"]
    nb.train(texts,labels)
    if args.predict:
        pred,scores=nb.predict(args.predict)
        print(json.dumps({"text":args.predict,"prediction":pred,"scores":scores}))
    else:
        correct=sum(1 for t,l in zip(texts,labels) if nb.predict(t)[0]==l)
        print(json.dumps({"training_accuracy":round(correct/len(texts),4),"vocab_size":len(nb.vocab),"classes":dict(nb.class_counts)},indent=2))
if __name__=="__main__":main()
