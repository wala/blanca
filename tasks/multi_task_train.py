"""
This is an example how to train SentenceTransformers in a multi-task setup.
The system trains BERT on the AllNLI and on the STSbenchmark dataset.
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses, InputExample
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, BinaryClassificationEvaluator
from sentence_transformers.readers import *
import logging
from datetime import datetime
import os
import json
from sklearn.model_selection import train_test_split
import argparse
import random

evaluation_steps = 1000

max_size = 23000
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
# Configure the training
num_epochs = 10
batch_size = 32

#### /print debug information to stdout

hierarchy_str = 'hierarchy'
linked_posts_str = 'linked_posts'
class_posts_str = 'class_posts'
posts_rank_str = 'posts_rank'
usage_str = 'usage'
search_str = 'search'

all_str = hierarchy_str + ',' + linked_posts_str + ',' + class_posts_str + ',' + posts_rank_str + ',' + usage_str + \
          ',' + 'search_str'


def create_hirerachy_examples(fl, data_dir, model, validate=None, is_test=False):
    train_hierarchy_samples = []
    disbn = []
    with open(os.path.join(data_dir, fl)) as f:
        data = json.load(f)
        max_distance = 0
        for obj in data:
            if obj['distance'] > max_distance:
                max_distance = obj['distance']
        for obj in data:
            # flip the meaning of similarity, since the more distant the two classes, the closer to 0 it should be
            dist = (max_distance - obj['distance']) / (max_distance - 1)
            train_hierarchy_samples.append(InputExample(texts=[obj['class1'], obj['class2']], label=dist))
            disbn.append(obj['distance'])
    random.shuffle(train_hierarchy_samples)
    train_hierarchy_samples = train_hierarchy_samples[:100000]
    disbn = disbn[:100000]

    if max_size:
        train_hierarchy_samples = train_hierarchy_samples[:max_size]
        disbn = disbn[:max_size]

    if is_test:
        return train_hierarchy_samples

    evaluator = None

    if hierarchy_str == validate:
        train_hierarchy_samples, dev_hierarchy_samples = train_test_split(train_hierarchy_samples, stratify=disbn, test_size=0.1)
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_hierarchy_samples, name='hierarchy')


    warmup_steps = math.ceil(len(train_hierarchy_samples) * num_epochs / batch_size * 0.1)  # 10% of train data for warm-up
    
    train_data_hierarchy = SentencesDataset(train_hierarchy_samples, model=model)
    train_dataloader_hierarchy = DataLoader(train_data_hierarchy, shuffle=True, batch_size=batch_size)
    train_loss_hierarchy = losses.CosineSimilarityLoss(model=model)

    print('H: Number of training examples: ', len(train_hierarchy_samples))

    global evaluation_steps
    evaluation_steps = math.ceil(len(train_hierarchy_samples) / 0.1)
    return train_dataloader_hierarchy, train_loss_hierarchy, evaluator, warmup_steps


def create_linked_posts(fl, data_dir, model, validate=None, is_test=False):
    train_linked_posts = []
    disbn = []

    with open(os.path.join(data_dir, fl)) as f:
        data = json.load(f)
        for obj in data:
            if obj['class'] == 'relevant':
                label = 1
            else:
                label = 0
            disbn.append(label)

            train_linked_posts.append(InputExample(texts=[obj['text_1'], obj['text_2']], label=label))
    random.shuffle(train_linked_posts)

    if is_test:
        return train_linked_posts

    if max_size:
        train_linked_posts = train_linked_posts[:max_size]

    evaluator = None
    if linked_posts_str == validate:
        train_linked_posts, dev_linked_posts = train_test_split(train_linked_posts, stratify=disbn, test_size=0.1)
        evaluator = BinaryClassificationEvaluator.from_input_examples(dev_linked_posts, name='linked-posts')

    warmup_steps = math.ceil(len(train_linked_posts) * num_epochs / batch_size * 0.1)  # 10% of train data for warm-up

    train_data_linked_posts = SentencesDataset(train_linked_posts, model=model)
    train_dataloader_linked_posts = DataLoader(train_data_linked_posts, shuffle=True, batch_size=batch_size)
    train_loss_linked_posts = losses.ContrastiveLoss(model=model)

    print('L: Number of training examples: ', len(train_linked_posts))

    global evaluation_steps
    evaluation_steps = math.ceil(len(train_linked_posts) / 0.1)

    return train_dataloader_linked_posts, train_loss_linked_posts, evaluator, warmup_steps


def create_train_class_posts(fl, data_dir, model, validate=None, is_test=False):
    train_class_posts = []
    disbn = []
    with open(os.path.join(data_dir, fl)) as f:
        data = json.load(f)
        for obj in data:
            train_class_posts.append(InputExample(texts=[obj['docstring'], obj['text']], label=obj['label']))
            disbn.append(obj['label'])
    random.shuffle(train_class_posts)

    if is_test:
        return train_class_posts
    if max_size:
        train_class_posts = train_class_posts[:max_size]

    evaluator = None
    if class_posts_str == validate:
        train_class_posts, dev_class_posts = train_test_split(train_class_posts, stratify=disbn, test_size=0.1)
        evaluator = BinaryClassificationEvaluator.from_input_examples(dev_class_posts, name='class-posts')
    warmup_steps = math.ceil(len(train_class_posts) * num_epochs / batch_size * 0.1)  # 10% of train data for warm-up


    train_data_class_posts = SentencesDataset(train_class_posts, model=model)
    train_dataloader_class_posts = DataLoader(train_data_class_posts, shuffle=True, batch_size=batch_size)
    train_loss_class_posts = losses.ContrastiveLoss(model=model)

    print('class_posts: Number of training examples: ', len(train_class_posts))

    global evaluation_steps
    evaluation_steps = math.ceil(len(train_class_posts) / 0.1)

    return train_dataloader_class_posts, train_loss_class_posts, evaluator, warmup_steps


def create_train_usage(fl, data_dir, model, validate=None, is_test=False):
    train_usage = []
    with open(os.path.join(data_dir, fl)) as f:
        data = json.load(f)
        min_d = 10000000
        max_d = 0
        for obj in data:
            dist = obj['distance']
            if dist < min_d:
                min_d = dist
            if dist > max_d:
                max_d = dist
        for obj in data:
            dist = (max_d - obj['distance']) / (max_d - min_d)
            train_usage.append(InputExample(texts=[obj['class1'], obj['class2']], label=dist))

    random.shuffle(train_usage)

    if is_test:
        return train_usage

    if max_size:
        train_usage = train_usage[:max_size]

    evaluator = None

    if usage_str == validate:
        train_usage, dev_usage = train_test_split(train_usage, test_size=0.1)
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_usage, name='usage')
    warmup_steps = math.ceil(len(train_usage) * num_epochs / batch_size * 0.1)  # 10% of train data for warm-up

    train_data_usage = SentencesDataset(train_usage, model=model)
    train_dataloader_usage = DataLoader(train_data_usage, shuffle=True, batch_size=batch_size)
    train_loss_usage = losses.CosineSimilarityLoss(model=model)

    print('U: Number of training examples: ', len(train_usage))

    global evaluation_steps
    evaluation_steps = math.ceil(len(train_usage) / 0.1)

    return train_dataloader_usage, train_loss_usage, evaluator, warmup_steps


def create_posts_ranking(fl, data_dir, model, validate=None, is_test=False):
    train_posts_ranking = []
    disbn = []
    with open(os.path.join(data_dir, fl)) as f:
        data = json.load(f)
        for obj in data:
            answers = obj['answers']
            filtered_answers = []
            votes = 1000000
            for answer in answers:
                my_votes = answer['a_votes']
                if my_votes < votes:
                    votes = my_votes
                    filtered_answers.append(answer)

            if len(filtered_answers) > 1:
                rank = len(filtered_answers)
                for answer in filtered_answers:
                    dist = rank / len(filtered_answers)
                    disbn.append(answer['a_rank'])
                    rank = rank - 1
                    train_posts_ranking.append(
                        InputExample(texts=[obj['q_text'], answer['a_text']], label=dist))

    random.shuffle(train_posts_ranking)

    print("data size " + str(len(train_posts_ranking)))
    
    if is_test:
        return train_posts_ranking

    if max_size:
        train_posts_ranking = train_posts_ranking[:max_size]

    evaluator = None
    if posts_rank_str == validate:
        train_posts_ranking, dev_posts_ranking = train_test_split(train_posts_ranking, test_size=0.1)
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_posts_ranking, name='posts ranking')

    warmup_steps = math.ceil(len(train_posts_ranking) * num_epochs / batch_size * 0.1)  # 10% of train data for warm-up

    train_data_posts_ranking = SentencesDataset(train_posts_ranking, model=model)
    train_dataloader_posts_ranking = DataLoader(train_data_posts_ranking, shuffle=True, batch_size=batch_size)
    train_loss_posts_ranking = losses.CosineSimilarityLoss(model=model)

    print('R: Number of training examples: ', len(train_posts_ranking))

    global evaluation_steps
    evaluation_steps = math.ceil(len(train_posts_ranking) / 0.1)

    return train_dataloader_posts_ranking, train_loss_posts_ranking, evaluator, warmup_steps


def create_search(collection, query_file, train, data_dir, model, validate=None, is_test=False):
    corpus = {}
    with open(os.path.join(data_dir, collection), 'r', encoding='utf8') as fIn:
        for line in fIn:
            pid, passage = line.strip().split("\t")
            corpus[pid] = passage

    queries = {}
    with open(os.path.join(data_dir, query_file), 'r', encoding='utf8') as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            queries[qid] = query

    train_search = []
    disbn = []
    with open(os.path.join(data_dir, train), 'r', encoding='utf8') \
            as f:
        added_q = set()
        for line in f.readlines():
            qid, pos_id, neg_id = line.strip().split()
            query = queries[qid]
            passage = corpus[pos_id]
            neg_passage = corpus[neg_id]
            if qid not in added_q:
                train_search.append(InputExample(texts=[query, passage], label=1))
                disbn.append(1)
                added_q.add(qid)
            train_search.append(InputExample(texts=[query, neg_passage], label=0))
            disbn.append(0)
    random.shuffle(train_search)

    if is_test:
        return train_search

    if max_size:
        train_search = train_search[:max_size]
    evaluator = None

    if search_str == validate:
        train_search, dev_search = train_test_split(train_search, stratify=disbn, test_size=0.1)
        evaluator = BinaryClassificationEvaluator.from_input_examples(dev_search, name='search')

    warmup_steps = math.ceil(len(train_search) * num_epochs / batch_size * 0.1)  # 10% of train data for warm-up

    # We create a DataLoader to load our train samples
    train_dataloader_search = DataLoader(train_search, shuffle=True, batch_size=batch_size)
    train_loss_search = losses.ContrastiveLoss(model=model)

    print('S: Number of training examples: ', len(train_search))

    global evaluation_steps
    evaluation_steps = math.ceil(len(train_search) / 0.1)

    return train_dataloader_search, train_loss_search, evaluator, warmup_steps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='multi task train')
    parser.add_argument('--model_name', type=str,
                        help='starting model name')
    parser.add_argument('--model_save_path', type=str,
                        help='where to save the model')
    parser.add_argument('--data_dir', type=str,
                        help='dir for all data needed to train')
    parser.add_argument('--tasks', type=str,
                        help='list of tasks to use to train:' + all_str)
    parser.add_argument('--validate', type=str,
                        help='validation task', required=False)
    parser.add_argument('--finetuned', action="store_true",
                        help='validation task', required=False)

    args = parser.parse_args()

    model_name = args.model_name

    model_save_path = args.model_save_path + model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if args.finetuned:
        word_embedding_model = models.Transformer(model_name, max_seq_length=256)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    else:
        # Use BERT for mapping tokens to embeddings
        word_embedding_model = models.Transformer(model_name)

        # Apply mean pooling to get one fixed sized sentence vector
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    train_objectives = []

    evaluator = None
    warmup_steps = 0
    # task 1 - class hierarchy prediction
    if hierarchy_str in args.tasks:
        train_dataloader_hierarchy, train_loss_hierarchy, e, w = create_hirerachy_examples('hierarchy_train.json', args.data_dir, model, validate=args.validate)
        train_objectives.append((train_dataloader_hierarchy, train_loss_hierarchy))

        if args.validate == hierarchy_str:
            warmup_steps = w
            evaluator = e

    # task 2 - determine if two posts are linked
    if linked_posts_str in args.tasks:
        train_dataloader_linked_posts, train_loss_linked_posts, e, w = create_linked_posts('stackoverflow_data_linkedposts__train.json', args.data_dir, model, validate=args.validate)
        train_objectives.append((train_dataloader_linked_posts, train_loss_linked_posts))

        if args.validate == linked_posts_str:
            warmup_steps = w
            evaluator = e

    # task 3 - determine if a post is related to a class's docstring
    if class_posts_str in args.tasks:
        train_dataloader_class_posts, train_loss_class_posts, e, w = create_train_class_posts('class_posts_train_data_new', args.data_dir, model, validate=args.validate)
        train_objectives.append((train_dataloader_class_posts, train_loss_class_posts))

        if args.validate == class_posts_str:
            warmup_steps = w
            evaluator = e

    # task 4 - class usage prediction
    if usage_str in args.tasks:
        train_dataloader_usage, train_loss_usage, e, w = create_train_usage('usage_train.json', args.data_dir, model, validate=args.validate)
        train_objectives.append((train_dataloader_usage, train_loss_usage))

        if args.validate == usage_str:
            warmup_steps = w
            evaluator = e

    # task 5 - predict ranks of a post's answers
    if posts_rank_str in args.tasks:
        train_dataloader_posts_ranking, train_loss_posts_ranking, e, w = create_posts_ranking('stackoverflow_data_ranking_v3_training.json', args.data_dir, model, validate=args.validate)
        train_objectives.append((train_dataloader_posts_ranking, train_loss_posts_ranking))

        if args.validate == posts_rank_str:
            warmup_steps = w
            evaluator = e

    # task 6 - predict search rank of a post
    if search_str in args.tasks:
        train_dataloader_search, train_loss_search, e, w = create_search('stackoverflow_matches_codesearchnet_5k_train_collection.tsv',
                             'stackoverflow_matches_codesearchnet_5k_train_queries.tsv',
                             'stackoverflow_matches_codesearchnet_5k_train_blanca-qidpidtriples.train.tsv',
                             args.data_dir, model, validate=args.validate)
        train_objectives.append((train_dataloader_search, train_loss_search))

        if args.validate == search_str:
            warmup_steps = w
            evaluator = e

    assert len(train_objectives) >= 1
    assert warmup_steps > 0

    # Train the model
    model.fit(train_objectives=train_objectives,
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=evaluation_steps,
          warmup_steps=warmup_steps,
          output_path=model_save_path
          )



  
