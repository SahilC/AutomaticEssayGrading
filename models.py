from sklearn import svm
import unicodecsv
import numpy as np
import codecs
import util

from Essay import Essay
from Kappa import get_average_kappa

# Trains the SVR classifier on training data and returns the classifier
def train_svr(training_data):
    feature_vector = []
    scores = []
    fo = codecs.open(training_data, encoding='utf-8')
    lines = fo.readlines()
    fo.close()
    
    line = 0
    for each_line in lines:
        row = each_line.split('\n')[0].split('\t')
        vector = []
        # Ignore the heading line
        if line < 1:
            line += 1
            continue
        if line % 50 == 0:
            print('Training sample: '+str(line))
        e = Essay(row, store_score = True)
        f = e.features
        for i in sorted(f.__dict__.keys()):
            vector.append(f.__dict__[i])
        scores.append(e.score)
        feature_vector.append(vector)
        line += 1
    clf = svm.SVR()
    clf.fit(np.array(feature_vector), np.array(scores))
    return clf

# Predicts scores for given data and returns the average quadratic kappa  
def predict_svr(clf, data):
    feature_vector = []
    scores = []
    predictions = []
    fo = codecs.open(data, encoding='utf-8')
    lines = fo.readlines()
    fo.close()
    
    line = 0
    for each_line in lines:
        row = each_line.split('\n')[0].split('\t')
        vector = []
        # Ignore the heading line
        if line < 1:
            line += 1
            continue
        if line % 50 == 0:
            print('Validation sample: '+str(line))
        e = Essay(row, store_score = True)
        f = e.features
        for i in sorted(f.__dict__.keys()):
            vector.append(f.__dict__[i])
        scores.append(e.score)
        feature_vector.append(vector)
        line += 1
                    
    predictions = clf.predict(np.array(feature_vector))
    return get_average_kappa(np.array(scores), np.array(predictions))

if __name__ == '__main__':
    training_data = 'dataset/small_training_set.tsv'
    test_data = 'dataset/small_test_set.tsv'
    classifier_dump = 'dumps/classifier_dump'
    
    print('\n----------------Training started----------------\n')
    svr_classifier = train_svr(training_data)
    print('\n----------------Training completed----------------\n')
    
    util.dump_classifier(svr_classifier, classifier_dump)
    
    print('\n----------------Testing started----------------\n')
    avg_kappa = predict_svr(svr_classifier, test_data)
    print('\n----------------Testing completed----------------\n')
    print('Average quadratic kappa : ' + str(avg_kappa))