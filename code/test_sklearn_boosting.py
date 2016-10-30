from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from Kappa import get_average_kappa
import util

def train_sklearn_boosting(training_data_dump):
    training_data = util.load_object(training_data_dump)
    model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=1), n_estimators=20)
    #model = AdaBoostRegressor(SVR(kernel='linear'), n_estimators=20)
    #model = RandomForestRegressor(n_estimators = 50)
    model = model.fit(training_data[:,:-1], training_data[:,-1])
    return model

def predict(model, test_data_dump):
    test_data = util.load_object(test_data_dump)
    predictions = []
    targets = []

    targets = test_data[:, -1]
    predictions = model.predict(test_data[:,:-1])
    return get_average_kappa(targets, predictions)

if __name__ == '__main__':
    training_data_dump = '../dumps/glove_training_data_dump'
    test_data_dump = '../dumps/glove_test_data_dump'

    print('--------------------sklearn adaboost regressor--------------------')
    model = train_sklearn_boosting(training_data_dump)
    avg_kappa = predict(model, test_data_dump)
    print('Average quadratic kappa : ' + str(avg_kappa))
