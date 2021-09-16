from utils import util
import argparse
import networkx as nx
from metrics_eval import ranking_metrics
import numpy as np
import scipy
import hierarchy_stats
import pickle
import sys

def sample_related_classes(class2superclasses, valid_classes):
    mod2classes = {}
    with open(class2superclasses) as f:
        for line in f:
            arr = line.split(',')
            clazz = arr[0].strip()
            if len(arr[1]) < len('http://purl.org/twc/graph4code/python/'):
                continue
            superclazz = arr[1][len('http://purl.org/twc/graph4code/python/'):].strip()
            if clazz not in valid_classes or superclazz not in valid_classes:
                continue
            module = clazz.split('.')[0]
            if superclazz.split('.')[0] == module:
                if module not in mod2classes:
                    mod2classes[module] = []
                mod2classes[module].append((clazz, superclazz))

    len_to_relation = {}
    for module in mod2classes:
        classgraph = nx.Graph()
        for edge in mod2classes[module]:
            clazz = edge[0]
            superclazz = edge[1]
            if clazz not in classgraph.nodes():
                classgraph.add_node(clazz)
            if superclazz not in classgraph.nodes():
                classgraph.add_node(superclazz)
            if superclazz != 'object':
                classgraph.add_edge(clazz, superclazz)
        print('starting sp computation:' + module)
        sp = dict(nx.all_pairs_shortest_path(classgraph, cutoff=10))
        
        for source in sp:
            for target in sp[source]:
                if source == target:
                    continue
                dist = len(sp[source][target]) - 1
                if dist not in len_to_relation:
                    len_to_relation[dist] = []
                len_to_relation[dist].append((source, target))

    with open('shortest_paths.pickle', 'wb') as handle:
        pickle.dump(len_to_relation, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return len_to_relation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hierarchy prediction based on embeddings')
    parser.add_argument('--top_k', type=int,
                        help='file containing all queries to be run')
    parser.add_argument('--class2superclass_file', type=str,
                        help='csv of classes to superclasses')
    parser.add_argument('--docstrings_file', type=str,
                        help='docstrings for classes, functions etc')
    parser.add_argument('--classmap', type=str,
                        help='classes to real class names as determined by dynamic loading of class')
    parser.add_argument('--classfail', type=str,
                        help='classes that fail to load to determine real class mappings')
    parser.add_argument('--embed_type', type=str,
                        help='USE or bert or roberta')
    parser.add_argument('--cache_sp', type=str,
                        help='shortest path computation cache pickle file')

    args = parser.parse_args()

    real_classes = hierarchy_stats.read_valid_classes(args.classmap, args.classfail)
    index, docList, docsToClasses, embeddedDocText, classesToDocs, doc2embeddings = util.build_index_docs(args.docstrings_file, args.embed_type, generate_dict=True, valid_classes=real_classes)
    if not args.cache_sp:
        len_related_classes = sample_related_classes(args.class2superclass_file, real_classes)
    else:
        with open(args.cache_sp, 'rb') as f:
            len_related_classes = pickle.load(f)

    # try finding embeddings for all path lengths in the file
    distances = []
    lengths = []
    for length in len_related_classes:
        len_distance = []
        for path in len_related_classes[length]:
            source = path[0]
            target = path[1]
            if source not in classesToDocs or target not in classesToDocs:
#                print('source or target not in classdocs:' + source + '->' + target)
                continue
            source_embedding = doc2embeddings[classesToDocs[source]].reshape(1,-1)
            target_embedding = doc2embeddings[classesToDocs[target]].reshape(1,-1)
        
            dist = scipy.spatial.distance.cdist(source_embedding, target_embedding, metric='cosine')
            lengths.append(length)
            distances.append(dist[0][0])
            len_distance.append(dist[0][0])
        with open(str(length) + '_distance.txt', 'w') as f:
            for x in len_distance:
                f.write(str(x) + '\n')
            f.close()

    print('sample size')
    print(len(distances))
    corr, p_value = scipy.stats.pearsonr(lengths, distances)
    print('correlation:' + str(corr))
    print('p-value:' + str(p_value))
