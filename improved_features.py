import tensorflow as tf
import numpy as np
import codecs
import time
import enchant
from Essay import Essay
from pymongo import MongoClient
from Kappa import get_average_kappa

d = enchant.Dict("en_US")

def predict_svr(clf, data):
    feature_vector = []
    scores = []
    predictions = []
    fo = codecs.open(data, encoding='utf-8')
    lines = fo.readlines()
    fo.close()
    client = MongoClient('localhost', 27017)
    db = client['nlprokz']
    glove = db.glove

    line = 0
    for each_line in lines:
        row = each_line.split('\n')[0].split('\t')
        essay_set = int(row[1])
        scores.append(get_score(essay_set,row))
        words = [i for i in each_line.lower().split()]
        word_list = []
        for word in words:
            if not d.check(word):
                suggest = d.suggest(word)
                if(len(suggest) > 0):
                    word_list.append(suggest[0])
            else:
                word_list.append(word)
        essay_vector = np.array([0.0 for i in xrange(300)])
        for i in glove.find({"gram":{"$in":word_list}}):
            word_vector = np.array([float(n) for n in i['glove_vector']])

            if(len(word_vector) == 300):
                essay_vector += word_vector
        feature_vector.append(essay_vector/len(words))
    # for each_line in lines:
    #     row = each_line.split('\n')[0].split('\t')
    #     vector = []
    #     # Ignore the heading line
    #     if line < 1:
    #         line += 1
    #         continue
    #     if line % 50 == 0:
    #         print('Validation sample: '+str(line))
    #     e = Essay(row, store_score = True)
    #     f = e.features
    #     for i in sorted(f.__dict__.keys()):
    #         vector.append(f.__dict__[i])
    #     scores.append(e.score)
    #     feature_vector.append(vector)
    #     line += 1

    predictions = clf.predict(np.array(feature_vector))
    return get_average_kappa(np.array(scores), np.array(predictions))

def get_score(essay_set, row):
    score = 0
    score = float(row[6])
    if essay_set == 1:
        div = 12
    elif essay_set == 2:
        div = 5
    elif essay_set == 3:
        div = 3
    elif essay_set == 4:
        div = 3
    elif essay_set == 5:
        div = 4
    elif essay_set == 6:
        div = 4
    elif essay_set == 7:
        div = 25
    elif essay_set == 8:
        div = 50
    return score/div

def build_essay_model(collection,essay):
    words = [i for i in essay.lower().split()]
    essay_vector = np.array([0.0 for i in xrange(300)])
    word_list = []
    word_list = words
    # for word in words:
    #     if not d.check(word):
    #         suggest = d.suggest(word)
    #         if(len(suggest) > 0):
    #             word_list += suggest[0]
    #         else:
    #             print word
    #     else:
    #         word_list.append(word)

    for i in collection.find({"gram":{"$in":word_list}}):
        word_vector = [float(n) for n in i['glove_vector']]

        if(len(word_vector) == 300):
            essay_vector += word_vector
    return (essay_vector/len(words))


def train_model(X_train, y_train):
    #  = np.load(PREPROCESSED_DATA)
    # X_train, y_train, testX, testY = mnist.load_data(one_hot=True)
    # X_train = X_train.reshape([-1, 28, 28, 1])
    y_train.shape = (-1,1)
    num_samples, num_timesteps, input_dim = X_train.shape
    net = tflearn.input_data(shape=[None, num_timesteps, input_dim])
    net = tflearn.lstm(net, 128)
    net = tflearn.fully_connected(net, 1, activation='relu')
    net = tflearn.regression(net, optimizer='sgd',
                             loss='mean_square', name="regression_output")
    model = tflearn.DNN(net, tensorboard_verbose=2)
    model.fit(X_train, y_train, n_epoch=5000, validation_set=0.1, show_metric=True,snapshot_step=100)

if __name__ == '__main__':
    # train_model()
    client = MongoClient('localhost', 27017)
    db = client['nlprokz']
    glove = db.glove
    feature_vector = []
    scores = []
    fo = codecs.open('dataset/small_training_set.tsv', encoding='utf-8')
    lines = fo.readlines()
    fo.close()
    line = 0
    for each_line in lines:
        line += 1
        if line == 1:
            continue
        if line % 50 == 0:
            print "TRAINED"+str(line)
        row = each_line.split('\n')[0].split('\t')
        essay_set = int(row[1])
        scores.append([get_score(essay_set,row)])
        feature_vector.append(build_essay_model(glove,each_line))

    feature_vector = np.array(feature_vector)
    scores = np.array(scores)
    test_feature_vector = []
    test_scores = []
    fo = codecs.open('dataset/small_test_set.tsv', encoding='utf-8')
    lines = fo.readlines()
    fo.close()
    line = 0
    for each_line in lines:
        line += 1
        if line == 1:
            continue
        if line % 50 == 0:
            print "TEST"+str(line)
        row = each_line.split('\n')[0].split('\t')
        essay_set = int(row[1])
        test_scores.append([get_score(essay_set,row)])
        test_feature_vector.append(build_essay_model(glove,each_line))

    test_feature_vector = np.array(test_feature_vector)
    test_scores = np.array(test_scores)

    EPOCHS = 10000
    PRINT_STEP = 1000

    x_ = tf.placeholder(tf.float32, [None, feature_vector.shape[1]])
    y_ = tf.placeholder(tf.float32, [None, 1])

    #cell = tf.nn.rnn_cell.BasicRNNCell(num_units=feature_vector.shape[1])
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(300, state_is_tuple=True) for steps in range(3)], state_is_tuple=True)

    outputs, states = tf.nn.rnn(cell, [x_], dtype=tf.float32)
    outputs = outputs[-1]

    W = tf.Variable(tf.random_normal([feature_vector.shape[1], 1]))
    b = tf.Variable(tf.random_normal([1]))

    y = tf.matmul(outputs, W) + b
    #prediction = tf.nn.softmax(tf.matmul(outputs, W) + b)
    cost = tf.reduce_mean(tf.square(y - y_))
    #train_op = tf.train.RMSPropOptimizer(0.005, 0.2).minimize(cost)
    train_op = tf.train.AdamOptimizer().minimize(cost)
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in range(EPOCHS):
            sess.run(train_op, feed_dict={x_:feature_vector, y_:scores})
            if i % PRINT_STEP == 0:
                c = sess.run(cost, feed_dict={x_:feature_vector, y_:scores})
                print('training cost:', c)

        response = sess.run(y, feed_dict={x_:test_feature_vector})
        print(response)
        print(test_scores)
        # incorrect = sess.run(error,{data: data, target: target})
        # incorrect = sess.run(error, feed_dict={x_: feature_vector, y_: scores})
        # print feature_vector[feature_vector.shape[0] - 1]
        # print sess.run(prediction,feed_dict={x_: feature_vector})
        #print sess.run(prediction,{data: [[[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],[1],[0],[0],[1],[1],[0],[1],[1],[1],[0]]]})
        # print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
        sess.close()

    # data = tf.placeholder(tf.float32, [None, 20,1]) #Number of examples, number of input, dimension of each input
    # target = tf.placeholder(tf.float32, [None, 21])
    # num_hidden = 24
    # cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)
    # val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
    # val = tf.transpose(val, [1, 0, 2])
    # last = tf.gather(val, int(val.get_shape()[0]) - 1)
    # weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
    # bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
    # #prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
    # prediction = tf.contrib.learn.TensorFlowRNNRegressor(output, y)
    # cross_entropy = -tf.reduce_sum(target * tf.log(prediction))
    # optimizer = tf.train.AdamOptimizer()
    # minimize = optimizer.minimize(cross_entropy)
    # mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
    # error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
    #
    # init_op = tf.initialize_all_variables()
    # sess = tf.Session()
    # sess.run(init_op)
    #
    # batch_size = 10
    # no_of_batches = int(len(feature_vector)) / batch_size
    # epoch = 5000
    # for i in range(epoch):
    #     ptr = 0
    #     for j in range(no_of_batches):
    #         inp, out = feature_vector[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
    #         ptr+=batch_size
    #         sess.run(minimize,{data: inp, target: out})
    #     print "Epoch ",str(i)

    # clf = MLPRegressor(activation='tanh',learning_rate='adaptive',early_stopping= True,learning_rate_init=0.002)
    # print('STARTING........')
    # start_time = time.time()
    # clf.fit(np.array(feature_vector), np.array(scores))
    # print("--- %s seconds ---" % (time.time() - start_time))
    # print('\n----------------Training completed----------------\n')
    # test_data = 'dataset/small_test_set.tsv'
    # avg_kappa = predict_svr(clf, test_data)
    # print('\n----------------Testing completed----------------\n')
    # print('Average quadratic kappa : ' + str(avg_kappa))
    # for each_line in lines:
    #     row = each_line.split('\n')[0].split('\t')
    #     vector = []
    #     # Ignore the heading line
    #     if line < 1:
    #         line += 1
    #         continue
    #     if line % 50 == 0:
    #         print('Training sample: '+str(line))
    #     e = Essay(row, store_score = True)
    #     f = e.features
    #     for i in sorted(f.__dict__.keys()):
    #         vector.append([f.__dict__[i]])
    #     scores.append(e.score)
    #     feature_vector.append(vector)
    #     line += 1
    # feature_vector = np.array(feature_vector)
    # print feature_vector.shape
    # train_model(feature_vector,np.array(scores))
