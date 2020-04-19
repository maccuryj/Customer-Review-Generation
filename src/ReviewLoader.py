import os
import csv
import numpy
import unicodedata
from itertools import chain
from joblib import dump, load
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, IterableDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from ReviewUtils import ReviewUtils, ReviewDataset

class ProductReviews():
    """
    Product Review Data Class that provides helper function for building
    necessary processing resources.
    Provides review dataloader for easy integration with PyTorch.

    Attributes:
        utils (ReviewUtils):            Utility object that holds information and helper classes
        word2id (dict):                 word to index dictionary
        id2word (dict):                 index to word dictionary        
    """    

    def __init__(self, utils):
        if not isinstance(utils, ReviewUtils):        
            raise ValueError("Argument 'utils' should be a ReviewUtils object!")
        self.utils = utils
        self.word2id = {}
        self.id2word = {}

    def create_vocabulary(self, files, max_features=10000, tfidf=False, load_clusters=True,  cluster_label_filename='ClusterDict.joblib'):
        """
        Make use of Term Frequency and (if set) Inverse Document Frequency to create a vocabulary
        for use in the RNN. The encoding is then adjusted and a decoding
        dictionary is created.
        """
        
        if tfidf:
            vectorizer = TfidfVectorizer(max_features=max_features,  token_pattern=r"(?u)\b\w+\b")
        else:
            vectorizer = TfidfVectorizer(input='filename', max_features=max_features, use_idf=False,  token_pattern=r"(?u)\b\w+\b")

            f = [os.path.join(self.utils.data_folder, file) for file in files]
            words = vectorizer.fit_transform(f)
            self.word2id = vectorizer.vocabulary_
        
        self.adjust_encoding()          
        
        if load_clusters:
            cluster_labels = self.load_cluster_labels(cluster_label_filename) 
            k = len(set(cluster_labels.values()))  
            self._cluster_encodings(k)     
        self.create_decoding()

        return self.word2id

    def adjust_encoding(self):
        """
        Adjust the word encoding to include language tokens
        """
        if 0 in self.word2id.values():
            for word in self.word2id:
                if self.word2id[word] == 0:
                    self.word2id[word] = len(self.word2id)
        self.word2id["<PAD>"] = 0

        encoding_size = len(self.word2id)

        tokens = ["<UNK>", "<EOR>"]
        i = 0
        for tok in tokens:
            if tok not in self.word2id.keys():
                self.word2id[tok] = encoding_size + i
                i += 1

    def _cluster_encodings(self, k):
        """
        Add the cluster start tokens to the word encoding

        Args:
            k (int):                Number of clusters            
        """
        
        encoding_size = len(self.word2id)           

        for i in range(k):
            token = "<SOR " + str(i) + ">"
            if token not in self.word2id.keys():
                self.word2id[token] = encoding_size + i

    def create_decoding(self):
        """
        Create the word decoding dictionary from the encoding dictionary
        """
        for word in self.word2id:
            self.id2word[self.word2id[word]] = word

    def get_reviewloader(self, batch_size=2048, folder=None, files=None, cluster_labels=None, embedding_method='nnEmbedding', embedding_dim=256):
        """
        Creates PyTorch Dataloader for LSTM training

        Args:
            batch_size (int):               Batch size for DataLoader
            files (str []:)                 Files to be processed by the DataLoader
            embedding_method (str):         Input Embedding (one-hot or nn.Embedding)
        """
        if folder is None:
            folder = self.utils.data_folder
        dataset = ReviewDataset(folder, files, 'train')

        if embedding_method not in ['nnEmbedding', 'onehot']:
            raise ValueError("Invalid embedding_method argument")
        if embedding_method == 'onehot':
            embedding_dim = len(self.word2id)

        embedder = Embedder(embedding_method, len(self.word2id), embedding_dim)
        collator = Collator(self.word2id, embedder, cluster_labels)
        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

        return loader

    def save_vocab(self, filename):
        """
        Save a vocabulary on disk.

        Args:
            filename (str):                 Name of file to save the words
        """
        self.utils.save_to_disk(self.utils.resource_folder, filename, self.word2id)


    def load_vocab(self, filename):
        """
        Load a vocabulary from disk.

        Args:
            filename (str):                 Name of file containing the words
        """
        self.word2id = self.utils.load_from_disk(self.utils.resource_folder)
        self.create_decoding()

        return self.word2id

    def load_cluster_labels(self, filename='ClusterDict.joblib'):
        """
        Load a dictionary of cluster labels from disk.

        Args:
            filename (str):                 Name of file containing the dictionary
        """
        cluster_labels = self.utils.load_from_disk(self.utils.resource_folder, filename)

        return cluster_labels

class Embedder:
    """
    Class for transforming words into vectors in the specified way. 
    Supports either a one-hot encoding, or an embedding of words 
    using Pytorch's nn.Embedding
    """

    def __init__(self, method, dict_size, embedding_dim):
        """
        Params:
            num_embeddings: How many words to encode?
            embedding_dim: size of embedded input vector x_t
        """
        self.num_embeddings = dict_size  
        self.embedding_dim = embedding_dim
        self.method = method
        
        if method == "onehot":
            if dict_size != embedding_dim:
                raise ValueError("If using one-hot, dict_size must equal embed_dim!")
        
        if method == "nnEmbedding":
            self.embedding = nn.Embedding(dict_size, embedding_dim)
            self.embedding.weight.requires_grad=False

    def one_hot_embedding(self, x):
        x_hot = one_hot(x, num_classes=self.embedding_dim)
        return x_hot.float()

    def embed(self, x):
        """
        Parameters:
            x: 
        """
        if self.method == "onehot":
            return self.one_hot_embedding(x)
        elif self.method == "nnEmbedding":
            return self.embedding(x)


class Collator():
    
    def __init__(self, encoding, embedder, cluster_labels):
        """        
        Args:
            encoding (dict):                word2id dictionary
            embedder (Embedder):            instance of class Embedder
            cluster_labels (dict):          cluster label dictionary 
        """
        self.encoding = encoding
        self.dict_size = len(encoding)
        self.embedder = embedder
        self.cluster_labels = cluster_labels

    def __call__(self, batch):
        
        X = []
        X_len = []
        Y = []

        for i, f, line in batch:            
            # Represent the line (review) as a list of integers
            encoded_line = [self.encoding[word.lower()] if word.lower() in self.encoding.keys()
            else self.encoding["<UNK>"]
            for word in line.split(' ')]
            # Change cluster dictionary filename format
            start_tag = "<SOR " + str(self.cluster_labels[f[:-3] + 'npy - ' + str(i + 1)]) + ">"
            encoded_line.insert(0, self.encoding[start_tag])
            encoded_line.append(self.encoding["<EOR>"])

            # Count the number of <UNK> tokens in the encoded_line (=review). If there are too many
            # unkowns, don't include this review in training
            unkowns = list(filter(lambda x: x == self.encoding["<UNK>"], encoded_line))
            num_unkowns = len(unkowns)
            if num_unkowns >= 3:
                continue
                
            # Embed the inputs of the sequence x = [x1, ... xT]
            x = torch.LongTensor(encoded_line[:-1])
            x = self.embedder.embed(x)            

            # The labels y = [y1, ..., yT] don't have to be embedded for CrossEntropyLoss
            y = torch.LongTensor(encoded_line[1:])

            X.append(x)
            X_len.append(len(encoded_line[:-1]))
            Y.append(y)
        try:
            X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        except Exception:
            return -1
        Y_padded = pad_sequence(Y, batch_first=True, padding_value=0)  
        #X_packed = pack_padded_sequence(X_padded, X_len, batch_first=True, enforce_sorted=False)
        
        return X_padded, X_len, Y_padded
