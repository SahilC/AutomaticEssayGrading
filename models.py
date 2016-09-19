from sklearn import svm
from Features import Features
if __name__ == "__main__":
    f = Features("This is me. This is my life.  Woah, woah woahooooaaa.")
    print f.__dict__
