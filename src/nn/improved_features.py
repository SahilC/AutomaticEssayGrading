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
    for word in words:
        if not d.check(word):
            suggest = d.suggest(word)
            if(len(suggest) > 0):
                word_list += suggest[0]
            else:
                print word
        else:
            word_list.append(word)

    for i in collection.find({"gram":{"$in":word_list}}):
        word_vector = [float(n) for n in i['glove_vector']]

        if(len(word_vector) == 300):
            essay_vector += word_vector
    return (essay_vector/len(words))

if __name__ == '__main__':
    # train_model()
    client = MongoClient('localhost', 27017)
    db = client['nlprokz']
    glove = db.glove
    feature_vector = []
    scores = []
    valid_feature_vector = []
    valid_scores = []
    fo = codecs.open('../../dataset/small_training_set.tsv', encoding='utf-8')
    lines = fo.readlines()
    fo.close()
    line = 0
    for each_line in lines:
        line += 1
        if line == 1:
            continue
        if line < 2200:
            if line % 50 == 0:
                print "TRAINED"+str(line)
            row = each_line.split('\n')[0].split('\t')
            essay_set = int(row[1])
            #scores.append([get_score(essay_set,row)])
            #feature_vector.append(build_essay_model(glove,each_line))
            vector = []
            e = Essay(row, store_score = True)
            f = e.features
            for i in sorted(f.__dict__.keys()):
                vector.append(f.__dict__[i])
            scores.append([e.score])
            feature_vector.append(vector)
        else:
            if line % 50 == 0:
                print "Validation"+str(line)
            row = each_line.split('\n')[0].split('\t')
            essay_set = int(row[1])
            #valid_scores.append([get_score(essay_set,row)])
            #valid_feature_vector.append(build_essay_model(glove,each_line))
            vector = []
            e = Essay(row, store_score = True)
            f = e.features
            for i in sorted(f.__dict__.keys()):
                vector.append(f.__dict__[i])
            valid_scores.append([e.score])
            valid_feature_vector.append(vector)

    feature_vector = np.array(feature_vector)
    scores = np.array(scores)

    valid_feature_vector = np.array(valid_feature_vector)
    valid_scores = np.array(valid_scores)

    test_feature_vector = []
    test_scores = []
    fo = codecs.open('../../dataset/small_test_set.tsv', encoding='utf-8')
    lines = fo.readlines()
    fo.close()
    line = 0
    for each_line in lines:
        line += 1
        if line == 1:
            continue
        if line % 50 == 0:
            print "TEST"+str(line)
        if line % 100 == 0:
            break
        row = each_line.split('\n')[0].split('\t')
        essay_set = int(row[1])
        #test_scores.append([get_score(essay_set,row)])
        #test_feature_vector.append(build_essay_model(glove,each_line))
        vector = []
        e = Essay(row, store_score = True)
        f = e.features
        for i in sorted(f.__dict__.keys()):
            vector.append(f.__dict__[i])
        test_scores.append([e.score])
        test_feature_vector.append(vector)

    test_feature_vector = np.array(test_feature_vector)
    test_scores = np.array(test_scores)

    EPOCHS = 10000
    PRINT_STEP = 1000

    x_ = tf.placeholder(tf.float32, [None, feature_vector.shape[1]])
    y_ = tf.placeholder(tf.float32, [None, 1])

    # validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(test_feature_vector,test_scores,every_n_steps=50)

    #cell = tf.nn.rnn_cell.BasicRNNCell(num_units=feature_vector.shape[1])
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(9, state_is_tuple=True) for steps in range(1)], state_is_tuple=True)

    outputs, states = tf.nn.rnn(cell, [x_], dtype=tf.float32)
    outputs = outputs[-1]

    W = tf.Variable(tf.random_normal([feature_vector.shape[1], 1]))
    b = tf.Variable(tf.random_normal([1]))

    y = tf.matmul(outputs, W) + b
    #prediction = tf.nn.softmax(tf.matmul(outputs, W) + b)
    cost = tf.reduce_mean(tf.square(y - y_))
    #train_op = tf.train.RMSPropOptimizer(0.005, 0.2).minimize(cost)
    train_op = tf.train.AdamOptimizer().minimize(cost)
    is_increasing = lambda L: reduce(lambda a,b: b if a < b else 9999 , L)!=9999
    costs = []
    count = 0
    i = 0
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        while(True):
            sess.run(train_op, feed_dict={x_:feature_vector, y_:scores})
            if i % PRINT_STEP == 0:
                c = sess.run(cost, feed_dict={x_:valid_feature_vector, y_:valid_scores})
                print('training cost:', c)
                costs.append(c)
                if len(costs) > 3 and is_increasing(costs[-3:]) and count > 2:
                    break
                elif len(costs) > 3 and is_increasing(costs[-3:]):
                    count+=1
                    print count
                # else:
                #     count = 0
            i += 1
        response = sess.run(y, feed_dict={x_:test_feature_vector})
        print get_average_kappa(response,test_scores)
        # incorrect = sess.run(error,{data: data, target: target})
        # incorrect = sess.run(error, feed_dict={x_: feature_vector, y_: scores})
        # print feature_vector[feature_vector.shape[0] - 1]
        # print sess.run(prediction,feed_dict={x_: feature_vector})
        #print sess.run(prediction,{data: [[[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],[1],[0],[0],[1],[1],[0],[1],[1],[1],[0]]]})
        # print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
        sess.close()
