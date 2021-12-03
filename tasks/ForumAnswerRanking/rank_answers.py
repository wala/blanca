import json
import tensorflow_hub as hub
import numpy as np
from bs4 import BeautifulSoup
import sys
import re
import math, scipy
from sentence_transformers import SentenceTransformer
import random
import statistics
from scipy import stats as stat
import pickle
from utils.util import get_model
import os, argparse
from pathlib import Path
from scipy.spatial import distance
from tqdm import tqdm
sample = False

q_ans_embeddings = {}
ans_embeddings = {}
def embed_sentences(sentences, model, embed_type ):
    if embed_type == 'USE':
        sentence_embeddings = model.encode([sentences])
    else:
        sentence_embeddings = model.encode(sentences)
    return sentence_embeddings

def get_embedding(content, model, embed_type):
    if content not in q_ans_embeddings:
        q_ans_embeddings[content] = embed_sentences(content, model, embed_type)

    return q_ans_embeddings[content]

def evaluate_task(data_path, model, embed_type):
    all_questions = []
    with open(data_path, 'r', encoding="UTF-8") as data_file:
        encounteredPosts = set()
        data = json.load(data_file)
        i = 0
        for q_data in data:
            stackUrl = q_data['q_url']
            if stackUrl in encounteredPosts:
                continue
            else:
                encounteredPosts.add(stackUrl)
            all_questions.append(q_data)
            i += 1

        folder_name = '/tmp/stackoverflow_embed_'+embed_type
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)
        print("Calculating MRR with model", embed_type)
        evaluate_MRR(all_questions, model, embed_type)
        print("Calculating NDCG with model", embed_type)
        evaluate_NDCG(all_questions, model, embed_type)

def evaluate_MRR(data, model, embed_type):
    recipRanks = []
    if sample:
        data = data[:100]
    euclid_distances_to_best_answer = []
    euclid_distances_to_worst_answer = []
    cosine_distances_to_best_answer = []
    cosine_distances_to_worst_answer = []

    for question_data in tqdm(data):
        q_embedding = get_embedding(question_data['q_text'], model, embed_type)

        voteOrder = []
        distanceOrder = []
        valid = True
        answerCollection = question_data['answers']
        min_vote = float('inf')
        max_vote = float('-inf')
        min_vote_idx = -1
        max_vote_idx = -1

        for idx, answer in enumerate(answerCollection):
            answerVotes = int(answer['a_votes']) if answer['a_votes'] != '' else None
            if answerVotes:
                if answerVotes < min_vote:
                    min_vote = answerVotes
                    min_vote_idx = idx
                if answerVotes > max_vote:
                    max_vote = answerVotes
                    max_vote_idx = idx
        for idx, answer in enumerate(answerCollection):
            answerText = answer['a_text']
            answerVotes = int(answer['a_votes']) if answer['a_votes'] != '' else 0
            ans_embedding = get_embedding(answerText, model, embed_type)
            dist = np.linalg.norm(ans_embedding-q_embedding)**2
            cosine_dist = distance.cosine(ans_embedding, q_embedding)
            voteOrder.append((answerVotes, answerText))
            distanceOrder.append((dist, answerText))
            if idx == min_vote_idx:
                euclid_distances_to_worst_answer.append(dist)
                cosine_distances_to_worst_answer.append(cosine_dist)

            elif idx == max_vote_idx:
                euclid_distances_to_best_answer.append(dist)
                cosine_distances_to_best_answer.append(cosine_dist)

        if not voteOrder:
            continue
        voteOrder.sort()
        voteOrder.reverse()
        distanceOrder.sort()

        if len(voteOrder) != 1:
            correctAnswer = voteOrder[0][1]
            for x in range(0, len(distanceOrder)):
                rank = x + 1
                reciprocal = 1/rank
                if distanceOrder[x][1] == correctAnswer:
                    recipRanks.append(reciprocal)
                    break

    meanRecipRank = sum(recipRanks)/len(recipRanks)
    print('MRR: standard error of the mean ', stat.sem(recipRanks))
    print("Mean reciprocal rank is:", meanRecipRank)
    print(f"Average distance from question to best answer (highest votes): euclid = {statistics.mean(euclid_distances_to_best_answer)}, "
          f"cosine = {statistics.mean(cosine_distances_to_best_answer)}")
    print(f"Average distance from question to worst answer (lowest votes):euclid = {statistics.mean(euclid_distances_to_worst_answer)}, "
          f"cosine = {statistics.mean(cosine_distances_to_worst_answer)}")

def evaluate_NDCG(data, model, embed_type):
    coefficients = []
    if sample:
        data = data[:100]
    for question_data in tqdm(data):
        q_embedding = get_embedding(question_data['q_text'], model, embed_type)

        voteOrder = []
        distanceOrder = []
        voteMap = {}
        answerCollection = question_data['answers']
        for answer in answerCollection:
            answerText = answer['a_text']
            answerVotes = int(answer['a_votes']) if answer['a_votes'] != '' else 0
            ans_embedding = get_embedding(answerText, model, embed_type)
            dist = np.linalg.norm(ans_embedding-q_embedding)**2
            voteOrder.append((answerVotes, answerText))
            distanceOrder.append((dist, answerText))
            voteMap[answerText] = answerVotes

        if not voteOrder:
            continue
        voteOrder.sort()
        voteOrder.reverse()
        distanceOrder.sort()
        if len(voteOrder) != 1:
            i = 1
            workingDCG = 0
            for distanceAnswer in distanceOrder:
                rel = voteMap[distanceAnswer[1]]
                normal = math.log2(i+1)
                totalAdd = rel/normal
                workingDCG += totalAdd
                i += 1
            i = 1
            workingIDCG = 0
            for voteAnswer in voteOrder:
                rel = voteAnswer[0]
                normal = math.log2(i+1)
                totalAdd = rel/normal
                workingIDCG += totalAdd
                i += 1
            if workingIDCG != 0:
                nDCG = workingDCG/workingIDCG
                coefficients.append(nDCG)
    fullNDCG = sum(coefficients)/len(coefficients)
    print('NDCG: standard error of the mean ', stat.sem(coefficients))
    print("Average NDCG:", fullNDCG)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hierarchy prediction based on embeddings')
    parser.add_argument('--eval_file', type=str,
                        help='train/test file')
    parser.add_argument('--embed_type', type=str,
                        help='USE or bert or roberta or finetuned or bertoverflow')
    parser.add_argument('--model_dir', type=str,
                        help='dir for finetuned or bertoverflow models', required=False)

    args = parser.parse_args()

    model = get_model(args.embed_type, args.model_dir)
    evaluate_task(args.eval_file, model, args.embed_type)
