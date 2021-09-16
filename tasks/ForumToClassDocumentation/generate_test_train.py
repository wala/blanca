import sys
import json
import random
from bs4 import BeautifulSoup


with open(sys.argv[1]) as f:
    positives = json.load(f)

with open(sys.argv[2]) as f:
    negatives = json.load(f)

flattened_positives = []
id = 0

def flatten(d, result, label):
    for p in d:
        obj = d[p]
        docstring = obj['docstring']
        for post in obj['posts']:
            text = post['title'] + ' ' + post['text']
            for answer in post['answers']:
                text = text + ' ' + answer['answer_text']
            soup = BeautifulSoup(text)
            result.append({'id': id, 'docstring': docstring, 'text': soup.get_text(), 'label': label})

flatten(positives, flattened_positives, 1)

flattened_negatives = []

flatten(negatives, flattened_negatives, 0)

frac = 0.1

test_pos = set(random.sample(list(range(len(flattened_positives))), int(frac*len(flattened_positives))))
test_neg = set(random.sample(list(range(len(flattened_negatives))), int(frac*len(flattened_negatives))))

train_pos = [n for i,n in enumerate(flattened_positives) if i not in test_pos]
train_neg = [n for i,n in enumerate(flattened_negatives) if i not in test_neg]

test_positives = [n for i,n in enumerate(flattened_positives) if i in test_pos]
test_negatives = [n for i,n in enumerate(flattened_negatives) if i in test_neg]

train = []
train.extend(train_pos)
train.extend(train_neg)

test = []
test.extend(test_positives)
test.extend(test_negatives)

with open('class_posts_train_data', 'w') as f:
    json.dump(train, f, indent=4)

with open('class_posts_test_data', 'w') as f:
    json.dump(test, f, indent=4)

    
print(len(test))
print(len(train))
