import ijson
import tensorflow_hub as hub 
import faiss 
import numpy as np 
from bs4 import BeautifulSoup 
from sentence_transformers import SentenceTransformer, models 
import torch 
import pandas as pd
import json
from utils import util
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import scipy


embed = None 


class USEModel(object):
    def __init__(self):
        model_path = 'https://tfhub.dev/google/universal-sentence-encoder/4'
        self.model = hub.load(model_path)

    def encode(self, sentences, batch_size=None, show_progress_bar=None, convert_to_numpy=True):
        return self.model(sentences)


def evaluate_classification(embed_type, model_path, dataSetPath, text1, text2, true_label, true_value):
    model = get_model(embed_type, model_path)
    with open(dataSetPath, 'r', encoding="UTF-8") as data_file:
        data = json.load(data_file)
        df = pd.DataFrame(data)

        trues = []
        falses = []
        cos_distance = []

        labels = []

        for jsonObject in data:
            srcEmbed = embed_sentences([jsonObject[text1]], model, embed_type)
            dstEmbed = embed_sentences([jsonObject[text2]], model, embed_type)
            from scipy.spatial import distance
            linkedDist = distance.cosine(srcEmbed, dstEmbed)
            cos_distance.append(linkedDist)

            if jsonObject[true_label] == true_value:
                trues.append(linkedDist)
                labels.append(1)
            else:
                falses.append(linkedDist)
                labels.append(0)

        out_df = pd.DataFrame(labels, columns =['label'])

        out_df['embedding_cosine_distance'] = cos_distance
        out_df.to_csv(embed_type + '_test_with_embeddings_distances.csv')
        print(np.mean(np.asarray(trues)))
        print(np.mean(np.asarray(falses)))

        print('Total number of samples = ', len(data))
        print(scipy.stats.ttest_ind(trues, falses))



def evaluate_regression(f, docPath, embedType, model_dir=None):
    df = pd.DataFrame(json.load(f))
    (index, docList, docsToClasses, embeddedDocText, classesToDocs, docToEmbedding) = util.build_index_docs(docPath, embedType, generate_dict=True, model_dir=model_dir)
    df = df[df['class1'].isin(classesToDocs.keys())]
    df = df[df['class2'].isin(classesToDocs.keys())]
    df['embedding1'] = df['class1'].apply(lambda x: docToEmbedding[classesToDocs[x]])
    df['embedding2'] = df['class2'].apply(lambda x: docToEmbedding[classesToDocs[x]])
    embed1 = df['embedding1'].values
    embed2 = df['embedding2'].values
    distance = []
    for idx in range(len(embed1)):
        distance.append(scipy.spatial.distance.cosine(embed1[idx], embed2[idx]))

    model = linear_model.LinearRegression()
    new_df = df[['distance']]
    model.fit(new_df.iloc[:], distance)
    y_pred = model.predict(new_df.iloc[:])
    # The coefficients
    print('Coefficients: \n', model.coef_)
    print('Mean squared error: %.2f' % mean_squared_error(distance, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f' % r2_score(distance, y_pred))

    corr, p_value = scipy.stats.pearsonr(df['distance'].values, distance)
    print('correlation:' + str(corr))
    print('p-value:' + str(p_value))
    out_df = df[['class1','class2','distance']]
    out_df['embedding_cosine_distance'] = distance
    out_df.to_csv(embedType + '_test_with_embeddings_distances.csv')



def get_model(embed_type, local_model_path='/data/BERTOverflow'):
    global embed 
    if embed: 
        return embed 
    if embed_type == 'USE': 
        embed = USEModel()
    elif embed_type == 'bertoverflow' or embed_type == 'finetuned': 
        print('Loading model from: ', local_model_path) 
        word_embedding_model = models.Transformer(local_model_path, max_seq_length=256) 
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension()) 
        embed = SentenceTransformer(modules=[word_embedding_model, pooling_model]) 
    elif embed_type == 'bert': 
        model_path = 'bert-base-nli-stsb-mean-tokens' 
        embed = SentenceTransformer(model_path) 
    elif embed_type == 'roberta': 
        model_path = 'roberta-base-nli-stsb-mean-tokens'
        embed = SentenceTransformer(model_path)
    elif embed_type == 'distilbert': 
        model_path = 'distilbert-base-nli-stsb-wkpooling' 
        embed = SentenceTransformer(model_path)
    elif embed_type == 'distilbert_para':
        model_path = 'distilroberta-base-paraphrase-v1'
        embed = SentenceTransformer(model_path)
    elif embed_type == 'xlm':
        model_path = 'xlm-r-distilroberta-base-paraphrase-v1'
        embed = SentenceTransformer(model_path)
    elif embed_type == 'msmacro':
        model_path = 'msmarco-distilroberta-base-v2'
        embed = SentenceTransformer(model_path)
    if torch.cuda.is_available() and embed_type != 'USE':
        embed = embed.to('cuda') 
    return embed 

def embed_sentences(sentences, embed_type, model_dir=None):
    embed = get_model(embed_type, model_dir)
    sentence_embeddings = embed.encode(sentences)
    return sentence_embeddings

def build_index_docs(docPath, embedType, valid_classes=None, generate_dict=False, model_dir=None):
    classesToDocs = {}
    docsToClasses = {}
    embedList = {}

    with open(docPath, 'r') as data:
        jsonCollect = ijson.items(data, 'item')
        i = 0
        for jsonObject in jsonCollect:
            if 'class_docstring' not in jsonObject:
                continue
            className = jsonObject['klass']
            if valid_classes and className not in valid_classes:
                continue
            docStringText = jsonObject['class_docstring']

            soup = BeautifulSoup(docStringText, 'html.parser')
            for code in soup.find_all('code'):
                code.decompose()  # this whole block might be unnecessary
            docStringText = soup.get_text()

            if docStringText in docsToClasses:
                docClasses = docsToClasses[docStringText]

                if className in docClasses:
                    pass

                else:
                    docClasses.append(className)

            else:
                docsToClasses[docStringText] = [className]
                
            classesToDocs[className] = docStringText

    docList = np.array(list(docsToClasses.keys()))
    embeddedDocText = np.array(embed_sentences(docList, embedType, model_dir))
    faiss.normalize_L2(embeddedDocText)
    index = faiss.IndexFlatIP(len(embeddedDocText[0]))
    index.add(embeddedDocText)

    if generate_dict:
        doc2embedding = {}
        for index, doc in enumerate(docList):
            doc2embedding[doc] = embeddedDocText[index]
        return (index, docList, docsToClasses, embeddedDocText, classesToDocs, doc2embedding)
    else:
        return (index, docList, docsToClasses, embeddedDocText, classesToDocs)


def compute_neighbor_docstrings(query_neighbors, docList):
    docstringsToNeighbors = {}

    for docStringIndex, embeddedDocStringNeighbors in enumerate(query_neighbors):
        docString = docList[docStringIndex]

        i = 0
        neighborDocstrings = []
        for neighborDocStringIndex in embeddedDocStringNeighbors:
            if i != 0:
                neighborDocstrings.append(docList[neighborDocStringIndex])
            i = i + 1

        docstringsToNeighbors[docString] = neighborDocstrings
        
    return docstringsToNeighbors
