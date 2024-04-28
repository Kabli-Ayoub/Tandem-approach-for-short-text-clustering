
import scipy
import metrics
import hdbscan
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import argparse
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from sentence_transformers import SentenceTransformer
import numba
import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
import umap.umap_ as umap



def preprocessing(file_path):
    preprocessed_sentences = []
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    with open(file_path, 'r', encoding = 'utf-8') as file:
        for line in file:
            line= line.strip()
            line = line.translate(str.maketrans('','', string.punctuation))
            tokens = word_tokenize(line)
            tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
            preprocessed_sentence = ' '.join(tokens)
            preprocessed_sentences.append(preprocessed_sentence)
    return preprocessed_sentences


def doc_2_vec(model, preprocessed_sentences):
    model = SentenceTransformer(model)
    list_sentence = []
    # Sentences are encoded by calling model.encode()
    for sentence in preprocessed_sentences:
        vec = model.encode(sentence)
        list_sentence.append(vec)
    return list_sentence




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='search_snippets',
                            choices=['stackoverflow', 'biomedical', 'searchsnippets'])
    parser.add_argument('--model', default="paraphrase-MiniLM-L6-v2", 
                        choices=['paraphrase-MiniLM-L6-v2', 'all-mpnet-base-v2'],type=str )
    parser.add_argument('--dim_red', default='UMAP', choices=['ACP', 'UMAP', 'TSNE'])
    args = parser.parse_args()

    if args.dataset == 'searchsnippets':
        file_path = 'data\SearchSnippets\SearchSnippets.txt'
        mat_path = 'data/SearchSnippets/SearchSnippets-STC2.mat'
    elif args.dataset == 'stackoverflow':
        file_path = 'data\stackoverflow\stackoverflow.txt'
        mat_path = 'data\stackoverflow\StackOverflow.mat'
    elif args.dataset == 'Biomedical':
        file_path = 'data\Biomedical\Biomedical.txt'
        mat_path = 'data\Biomedical\Biomedical-STC2.mat'
    print("########## Execution has began ########## ")
    preprocessed_sentences = preprocessing(file_path)
    print("Preprocessing done!")
    list_sentence = doc_2_vec(args.model, preprocessed_sentences)
    print('Sentence embedding done! ')
    if args.dim_red == 'UMAP':
        umap_embeddings = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=100,random_state=2024).fit_transform(list_sentence)

        
    
    print('Dimension reduction done!')
    
    kmeans = KMeans(n_clusters=8, init='k-means++', n_init ='auto', random_state=2024)
    kmeans.fit_transform(umap_embeddings)
    print('Clustering  done!')
    
    y_pred = kmeans.labels_

    mat = scipy.io.loadmat(mat_path)
    y = np.squeeze(mat['labels_All'])

    print('acc:', metrics.acc(y, y_pred))
    print('nmi', metrics.nmi(y, y_pred))