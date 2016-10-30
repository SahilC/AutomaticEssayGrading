from sklearn import svm
import util
from Kappa import get_average_kappa

# Trains the SVR classifier on training data and returns the classifier
def train_svr(training_data_dump):
    training_data = util.load_object(training_data_dump)
    clf = svm.SVR(kernel = 'rbf')
    clf.fit(training_data[:,:-1], training_data[:,-1])
    return clf

# Predicts scores for given data and returns the average quadratic kappa  
def predict_svr(clf, test_data_dump):
    test_data = util.load_object(test_data_dump)
    targets = test_data[:,-1]
    predictions = clf.predict(test_data[:,:-1])
    return get_average_kappa(targets, predictions)

if __name__ == '__main__':
    training_data_dump = '../dumps/training_data_dump'
    test_data_dump = '../dumps/test_data_dump'
    
    print('--------------------sklearn SVR--------------------')
    model = train_svr(training_data_dump)
    avg_kappa = predict_svr(model, test_data_dump)
    print('Average quadratic kappa : ' + str(avg_kappa))