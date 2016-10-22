from pymongo import MongoClient
import glob
import os

if __name__ == '__main__':
    client = MongoClient('localhost', 27017)
    db = client['nlprokz']
    glove = db.glove
    count = 0
    for root, dirs, files in os.walk('Glove/'):
        for basename in files:
            filename = os.path.join(root, basename)
            bulk_grams = []
            for line in open(filename):
                grams = line.split()
                word = grams[0]
                rest = grams[1:]
                bulk_grams.append({"gram":word,"glove_vector":rest})
                if len(bulk_grams) > 5000:
                    count += 5000
                    print 'Inserted '+str(count)
                    glove.insert_many(bulk_grams)
                    bulk_grams = []
