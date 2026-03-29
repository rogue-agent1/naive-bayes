from naive_bayes import GaussianNB, MultinomialNB
X = [[1,1],[2,1],[1,2],[8,8],[9,8],[8,9]]
y = [0,0,0,1,1,1]
gnb = GaussianNB().fit(X, y)
assert gnb.predict_one([1.5, 1.5]) == 0
assert gnb.predict_one([8.5, 8.5]) == 1
preds = gnb.predict(X)
assert preds == y
mnb = MultinomialNB().fit([[1,0,1],[0,1,0],[1,1,0],[0,0,1]], ["a","b","a","b"])
assert mnb.predict_one([1,0,1]) == "a"
print("naive_bayes tests passed")
