import json
import sys
import pandas as pd
from scipy.spatial import distance
from utils.util import get_model, embed_sentences
import numpy as np
from scipy import stats

def evaluate_classification(embed_type, model_path, dataSetPath, otherDatasetPath):
    model = get_model(embed_type, model_path)
    ids = set()
    with open(dataSetPath, 'r', encoding="UTF-8") as data_file:
        data = json.load(data_file)
        for row in data:
            ids.add(row['url_1'].split('/')[-1])
            ids.add(row['url_2'].split('/')[-1])

    skipped = 0
    label2distance = {}

    with open(otherDatasetPath, 'r', encoding="UTF-8") as other:
        df = pd.read_csv(other)
#        df = df.sample(frac=0.05, replace=False, random_state=1)
        
        print(df.count)
        for _, row in df.iterrows():
            if row['q1_Id'] in ids or row['q2_Id'] in ids:
                skipped += 1
                continue
            srcEmbed = embed_sentences([row['q1_AnswersBody']], model, embed_type)
            dstEmbed = embed_sentences([row['q2_AnswersBody']], model, embed_type)
            linkedDist = distance.cosine(srcEmbed, dstEmbed)
            lbl = row['class'].strip().replace('"','')
            if lbl not in label2distance:
                label2distance[lbl] = []
            label2distance[lbl].append(linkedDist)

    print('skipped:' + str(skipped))
    
    for key in label2distance:
        print(key)
        print(np.mean(np.asarray(label2distance[key])))
        print(len(label2distance[key]))
    direct = label2distance['direct']
    duplicate = label2distance['duplicate']
    indirect = label2distance['indirect']
    isolated = label2distance['isolated']

    print('direct-isolated')
    print(stats.ttest_ind(direct, isolated))
    print('dup - isolated')
    print(stats.ttest_ind(duplicate, isolated))
    print('indirect - isolated')
    print(stats.ttest_ind(indirect, isolated))
    


if __name__ == '__main__':
    dataSetPath = sys.argv[1]
    other_datasetPath = sys.argv[2]
    embed_type = sys.argv[3]
    model_path = sys.argv[4]
    evaluate_classification(embed_type, model_path, dataSetPath, other_datasetPath)
