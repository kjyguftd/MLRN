import glob
import hashlib
import os

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class WeisfeilerLehmanMachine:
    """
    Weisfeiler Lehman feature extractor class.
    """

    def __init__(self, graph, features, iterations):
        """
        Initialization method which also executes feature extraction.
        :param graph: The Nx graph object.
        :param features: Feature hash table.
        :param iterations: Number of WL iterations.
        """
        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes = self.graph.nodes()
        self.extracted_features = [self.hash_feature(str(v)) for k, v in features.items()]
        self.do_recursions()

    def hash_feature(self, feature):
        hash_object = hashlib.md5(str(feature).encode())
        hash_float = int(hash_object.hexdigest(), 16) % (10 ** 8) / (10 ** 8)
        return hash_float

    def do_a_recursion(self):
        """
        The method does a single WL recursion.
        :return new_features: The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.nodes:
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node])] + sorted([str(deg) for deg in degs])
            features = "_".join(features)
            hash_object = hashlib.md5(features.encode())
            # hashing = hash_object.hexdigest()
            hashing = int(hash_object.hexdigest(), 16) % (10 ** 8) / (10 ** 8)
            new_features[node] = hashing
        self.extracted_features = [(prev + new) / 2 for prev, new in zip(self.extracted_features, new_features.values())]
        return new_features

    def do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.iterations):
            self.features = self.do_a_recursion()


def path2name(path):
    base = os.path.basename(path)
    name = os.path.splitext(base)[0].split('_')
    return name[1]



def dataset_reader(path):
    """
    Function to read the graph and features from a json file.
    :param path: The path to the graph json.
    :return graph: The graph object.
    :return features: Features hash table.
    :return name: Name of the graph.
    """
    name = path2name(path)
    data = pd.read_csv(path)
    graph = nx.from_pandas_edgelist(data, source='source', target='target')

    if 'node' in data.columns and 'feature' in data.columns:
        features = data[['node', 'feature']].set_index('node')['feature'].to_dict()
    else:
        features = dict(graph.degree())

    return graph, features, name


def feature_extractor(path, rounds=3):
    """
    Function to extract WL features from a graph.
    :param path: The path to the graph json.
    :param rounds: Number of WL iterations.
    :return doc: Document collection object.
    """

    graph, features, name = dataset_reader(path)

    machine = WeisfeilerLehmanMachine(graph, features, rounds)
    doc = TaggedDocument(words=machine.extracted_features, tags=["g_" + name])
    return doc, machine.extracted_features, name
    # return np.array(machine.extracted_features)


def main():
    """
    Main function to read the graph list, extract features.
    Learn the embedding.
    Use supervised learning and unsupervised learning for classification.
    """
    graphs = glob.glob(os.path.join('dataset\edges', "*.csv"))
    # could be replaced with 'dataset\edges_4', get results of networks with the threshold of 0.4

    labels_df = pd.read_csv('dataset\label1.csv')
    graph_sample = graphs


    print("\nFeature extraction started.\n")
    document_collections = Parallel(n_jobs=4)(
        delayed(feature_extractor)(g) for g in tqdm(graph_sample))
    documents = [doc for doc, _, _ in document_collections]
    labels = []
    for g in graph_sample:
        matching_labels = labels_df.loc[labels_df['graph_name'] == path2name(g), 'label'].values
        if len(matching_labels) > 0:
            labels.append(matching_labels[0])
        else:
            print(path2name(g))
    print("\nTraining Doc2Vec model.\n")
    model = Doc2Vec(documents, vector_size=128, window=5, min_count=1, workers=4, epochs=10)

    doc_vectors = np.array([model.dv[tag] for tag in model.dv.index_to_key])


    """
    Unsupervised Learning
    """
    # GMM
    n_components_range = np.arange(1, 21)

    bics = []
    aics = []

    for n in n_components_range:
        gmm = GaussianMixture(n_components=n, random_state=0)
        gmm.fit(doc_vectors)
        bics.append(gmm.bic(doc_vectors))
        aics.append(gmm.aic(doc_vectors))

    optimal_n_components = n_components_range[np.argmin(bics)]

    gmm = GaussianMixture(n_components=optimal_n_components, random_state=0)
    cluster_labels_gmm = gmm.fit_predict(doc_vectors)
    print("\nGMM Clustering Results:")

    # K-Mean
    pca = PCA(n_components=2)
    dov_vectors_pca = pca.fit_transform(doc_vectors)
    kmeans = KMeans(n_clusters=2, random_state=0)
    cluster_labels_kmeans = kmeans.fit_predict(dov_vectors_pca)
    print("\nK-Means Clustering Results: ")

    name = "K-Means"
    # could be replaced with "GMM", then you can get GMM results
    cluster_labels = cluster_labels_kmeans
    # could be replaced with cluster_labels_gmm, then you can get GMM results

    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(os.path.basename(graphs[i]))

    cluster_label_counts = {label: {} for label in clusters}
    for i, cluster in enumerate(cluster_labels):
        doc_label = labels[i]
        if doc_label not in cluster_label_counts[cluster]:
            cluster_label_counts[cluster][doc_label] = 0
        cluster_label_counts[cluster][doc_label] += 1

    for label, docs in clusters.items():
        print(f"\nCluster {label}:")
        for doc in docs:
            print(doc)
        print("label distribution:")
        total_docs = len(docs)
        print(f"Total number of graph: {total_docs}")
        for doc_label, count in cluster_label_counts[label].items():
            print(f"Label {doc_label}: {count}")

    doc_vectors_2d = TSNE(n_components=2, random_state=0).fit_transform(doc_vectors)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=doc_vectors_2d[:, 0], y=doc_vectors_2d[:, 1],
        hue=cluster_labels,
        palette=sns.color_palette("hsv", 8),
        legend="full"
    )
    plt.title(f"t-SNE visualization of {name} Clusters")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.show()


    """
    Supervised Learning
    """
    num_iteration = 10
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
    }

    for name, classifier in classifiers.items():
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []

        for _ in range(num_iteration):
            vectors = [model.dv[doc.tags[0]] for doc in documents]
            X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.2, random_state=42)

            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            accuracies.append(accuracy_score(y_test, y_pred))
            precisions.append(precision_score(y_test, y_pred, average='weighted'))
            recalls.append(recall_score(y_test, y_pred, average='weighted'))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

        print(f"{name} Model accuracy: {np.mean(accuracies)}")
        print(f"{name} Model precision: {np.mean(precisions)}")
        print(f"{name} Model recall: {np.mean(recalls)}")
        print(f"{name} Model F1-Score: {np.mean(f1_scores)}")


if __name__ == "__main__":
    main()