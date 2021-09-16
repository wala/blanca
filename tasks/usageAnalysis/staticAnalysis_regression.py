import sys
from utils import util
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import scipy
import json
import argparse
# embedType = 'USE'

def build_class_mapping(mapPath):
    classMap = {}
    with open(mapPath, 'r') as inputFile:
        for line in inputFile:
            lineComponents = line.rstrip().split(' ')
            if len(lineComponents) < 2:
                classMap[lineComponents[0]] = lineComponents[0]
            else:
                classMap[lineComponents[0]] = lineComponents[1]
    return classMap


def check(data, v):
    print(v)
    assert v in data
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hierarchy prediction based on embeddings')
    parser.add_argument('--eval_file', type=str,
                        help='train/test file')
    parser.add_argument('--embed_type', type=str,
                        help='USE or bert or roberta or finetuned or bertoverflow')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='dir for finetuned or bertoverflow models', required=False)
    parser.add_argument('--docstrings_file', type=str,
                        help='docstrings for classes, functions etc')
    parser.add_argument('--classmap',type=str,
                        help='class_map')

    args = parser.parse_args()

    # docPath = sys.argv[1]
    # classPath = sys.argv[2]
    # usagePath = sys.argv[3]
    # if len(sys.argv) > 4:
    #     embedType = sys.argv[4]
    #     if len(sys.argv) > 5:
    #         model_dir = sys.argv[5]
        
    util.get_model(args.embed_type, args.model_dir)

    with open(args.eval_file) as f:
        util.evaluate_regression(f, args.docstrings_file, args.embed_type, args.model_dir)
        

        
