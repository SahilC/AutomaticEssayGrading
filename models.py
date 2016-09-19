from sklearn import svm
from Features import Features
if __name__ == "__main__":
    feature_vector = []
    scores = []
    vector = []
    f = Features("This is me. This is my life.  Woah, woah woahooooaaa.")
    for i in sorted(f.__dict__.keys()):
        vector.append(f.i)

    feature_vector.append(vector)
    clf = svm.SVR()
    clf.fit(feature_vector, scores)
