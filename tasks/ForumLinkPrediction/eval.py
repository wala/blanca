from utils import util
import sys
import ijson
import numpy as np
import scipy
import json
import collections

embedType = 'USE'

ids = set()


def get_ids(testSetPath):
    with open(testSetPath, 'r') as testSet:
        for test in ijson.items(testSet, "item"):
            ids.add(int(test[0]))
            ids.add(int(test[1]))


def get_embeddings(dataSetPath):
    order = {}
    posts = []
    i = 0
    with open(dataSetPath, 'r') as dataSet:
        for post in ijson.items(dataSet, "item"):
            if int(post['id']) in ids:
                posts.append(post['text'])
                order[int(post['id'])] = i
                i = i + 1
        print('read posts')
        if len(posts) != len(set(posts)):
            print("dups found")
            post2counts = collections.Counter(posts)
            for post, count in post2counts.items():
                if count > 1:
                    print('--------------')
                    print(post)

    return order, util.embed_sentences(np.array(posts), embedType)


trues = []
falses = []


def check(testSetPath, order, embeddings):
    with open(testSetPath, 'r') as testSet:
        for test in ijson.items(testSet, "item"):
            srcEmbed = embeddings[order[int(test[0])]]
            dstEmbed = embeddings[order[int(test[1])]]

            linkedDist = np.linalg.norm(srcEmbed - dstEmbed) ** 2

            if test[2]:
                trues.append(linkedDist)
            else:
                falses.append(linkedDist)


if __name__ == '__main__':
    embedType = sys.argv[3]

    dataSetPath = sys.argv[1]
    testSetPath = sys.argv[2]
    embedType = sys.argv[3]

    print('calling get ids')
    get_ids(testSetPath)

    (order, embeddings) = get_embeddings(dataSetPath)

    check(testSetPath, order, embeddings)

    with open(sys.argv[4], 'w') as trueFile:
        trueFile.write(json.dumps(trues, indent=2))

    with open(sys.argv[5], 'w') as falseFile:
        falseFile.write(json.dumps(falses, indent=2))

    print(scipy.stats.ttest_rel(trues, falses))

