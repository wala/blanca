import sys
import re
import math
import random
import json
import numpy as np

def histedges_equalN(x, nbins):
    data_sorted = sorted(x, key=lambda y: y[0])
    step = math.ceil(len(data_sorted)//nbins+1)
    binned_data = []
    for i in range(0,len(data_sorted),step):
        binned_data.append(data_sorted[i:i+step])
    return binned_data

def capture_characteristics(sample_t):
    all_distance = np.asarray([i['distance'] for i in sample_t])
    print('mean distance:' + str(np.mean(all_distance)))
    print('min distance:' + str(np.amin(all_distance)))
    print('max distance:' + str(np.amax(all_distance)))

with open(sys.argv[1]) as f:
    staticData = f.readlines()
    
    matchString = '(.+) (\d+) \[(.+)\]'
    added_pairs = set()

    all_pairs = []
    idx = 0
    min_size = 500000
    max_shared_calls = 0
    for line in staticData:
        pattern = re.compile(matchString)
        adjustedLine = pattern.match(line)
        if adjustedLine == None:
            print("Found violation.")
            print(line)
        count = int(adjustedLine.group(2))
        if count > max_shared_calls:
            max_shared_calls = count
        klass = adjustedLine.group(1)
        otherClasses = adjustedLine.group(3).strip().split(', ')
        size = len(otherClasses)
        if size < min_size:
            min_size = size
        for c in otherClasses:
            p = [c, klass]
            p.sort()
            key = p[0] + '|' + p[1]
            if key in added_pairs:
                continue
            added_pairs.add(key)
            idx += 1
            all_pairs.append((p[0], p[1], count, size))

    # for every pair in all pairs, create a vector of count, size, and find euclidean distance from the
    # 'ideal' pair, which is the pair with the best shared call counts and the smallest size of other classes
    # it shares the call sequence with
    all_vector_distances = []
    base = np.asarray([max_shared_calls, min_size])
    new_all_pairs = []
    for idx, i in enumerate(all_pairs):
        a = np.asarray([i[2], i[3]])
        d = np.linalg.norm(a-base)
        all_vector_distances.append((d, idx))
        new_all_pairs.append({'id': idx, 'class1' : i[0], 'class2' : i[1], 'distance': d}) 
    all_in_test_sample = []
    
    bins = histedges_equalN(all_vector_distances, 10)
    test_idxs = []
    for bin in bins:
        n = math.ceil(len(bin)/10)
        sample = random.sample(bin, n)
        for i in sample:
            all_in_test_sample.append(new_all_pairs[i[1]])
            test_idxs.append(new_all_pairs[i[1]]['id'])

    print('test characteristics')    
    capture_characteristics(all_in_test_sample)
    print('train characteristics')
    train_sample = [i for i in new_all_pairs if i['id'] not in test_idxs]
    capture_characteristics(train_sample)
    
    with open(sys.argv[2], 'w') as f:
        json.dump(train_sample, f, indent=4)

    with open(sys.argv[3], 'w') as f:
        json.dump(all_in_test_sample, f, indent=4)
