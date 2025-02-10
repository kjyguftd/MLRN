import csv
import os

import networkx as nx
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def matrix_detection(texts, alpha=0.5):
    '''
    :param texts
    :param alpha: controls the blending ratio of TF-IDF and SentenceTransformer, default 0.5
    :return: combined similarity matrix and aggregated semantic feature
    '''
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    tfidf_similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    semantic_features = model.encode(texts)
    semantic_similarity_matrix = cosine_similarity(semantic_features)

    combined_similarity_matrix = alpha * tfidf_similarity_matrix + (1 - alpha) * semantic_similarity_matrix
    aggregate_semantic_feature = np.mean(semantic_features, axis=0)

    return combined_similarity_matrix, aggregate_semantic_feature


def create_graph_using_tfidf(cleaned_data, th):
    '''
    :param cleaned_data
    :param th: threshold that controls the formation of links
    :return: graph and aggregated semantic feature
    '''
    g = nx.Graph()
    nodes = []
    node_ids = cleaned_data.index.tolist()

    for index, row in cleaned_data.iterrows():
        g.add_node(index)
        nodes.append(row.iloc[0])

    similarity_matrix, aggregate_semantic_feature = matrix_detection(nodes)

    for index, vec in enumerate(similarity_matrix):
        vec[index] = -1
        for target, similarity in enumerate(vec):
            if similarity > th:
                g.add_edge(node_ids[index], node_ids[target], weight=similarity)

    return g, aggregate_semantic_feature


def construct_graph(source_folder, destination_folder, th = 0):
    '''
    :param source_folder
    :param destination_folder
    :param th: threshold that controls the formation of links, default 0
    '''
    aggregate_features_list = []

    for filename in os.listdir(source_folder):
        source_file = os.path.join(source_folder, filename)
        file_folder = os.path.splitext(filename)[0]

        ds_graph = pd.read_csv(source_file, encoding='utf-8').head(10000)
        ds_graph.dropna(subset=['cleaned_text'], inplace=True)
        g, aggregate_semantic_feature = create_graph_using_tfidf(ds_graph, th)
        aggregate_features_list.append(aggregate_semantic_feature)

        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        destination_edges = os.path.join(destination_folder, f'{file_folder}_edges.csv').replace('\\', '/')
        edges_df = pd.DataFrame(g.edges(data=True), columns=["source", "target", "data"])
        edges_df["weight"] = edges_df["data"].apply(lambda x: x["weight"])
        edges_df.drop(columns=["data"], inplace=True)
        edges_df.to_csv(destination_edges, index=False)

    destination_aggregate_features = os.path.join(destination_folder, 'aggregate_features4.csv').replace('\\', '/')
    pd.DataFrame(aggregate_features_list).to_csv(destination_aggregate_features, index=False)
