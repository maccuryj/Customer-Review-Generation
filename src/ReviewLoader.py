import os
import csv
import numpy
import pickle
import unicodedata
from itertools import chain
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, IterableDataset
from sklearn.feature_extraction.text import TfidfVectorizer

class ProductReviews():
    """
    Product Review Data Class that provides helper function for building
    necessary processing resources.
    Provides review dataloader for easy integration with PyTorch.

    Attributes:
        root_dir (str):                 absolute path of review directory
        word2id (dict):                 word to index dictionary
    """    

    def __init__(self, review_dir):        
        self.review_dir = review_dir              
        self.word2id = {}
        self.id2word = {}
        self.encoding_filename = ""

    def create_vocabulary(self, file, max_features=10000, tfidf=False):
        """
        Make use of Term Frequency and (if set) Inverse Document Frequency to create a vocabulary
        for use in the RNN. The encoding is then adjusted and a decoding
        dictionary is created.
        """
        if tfidf:
            vectorizer = TfidfVectorizer(max_features=max_features,  token_pattern=r"(?u)\b\w+\b")
        else:
            vectorizer = TfidfVectorizer(max_features=max_features, use_idf=False,  token_pattern=r"(?u)\b\w+\b")
        with open(os.path.join(self.review_dir, file), 'r') as f:
            words = vectorizer.fit_transform(f)
        self.word2id = vectorizer.vocabulary_
        
        self.adjust_encoding()
        self.create_decoding()        

        return vectorizer, words

    def adjust_encoding(self):
        """
        Adjust the word encoding to include language tokens
        """
        if 0 in self.word2id.values():
            for word in self.word2id:
                if self.word2id[word] == 0:
                    self.word2id[word] == len(self.word2id)
        self.word2id["<PAD>"] = 0

        encoding_size = len(self.word2id)

        tokens = ["<UNK>", "<SOR>", "<EOR>"]
        i = 0
        for tok in tokens:
            if tok not in self.word2id.values():
                self.word2id[tok] = encoding_size + i
                i += 1

    def create_decoding(self):
        """
        Create the word decoding dictionary from the encoding dictionary
        """
        for word in self.word2id:
            self.id2word[self.word2id[word]] = word

    def get_reviewloader(self, batch_size):
        """
        Creates PyTorch Dataloader for reviews

        Args:
            batch_size (int):               Batch size for DataLoader
        """
        collator = Collator(self.word2id)
        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

        return loader

    def save_vocab(filename):
        """
        Save a vocabulary on disk.

        Args:
            filename (str):                 Name of file to save the words
        """
        if '.pickle' not in filename:
                filename = filename + '.pickle'

        pickle.dump(word2id, open(os.path.join(review_dir, filename)))
        self.encoding_filename = filename


    def load_vocab(filename):
        """
        Load a vocabulary from disk.

        Args:
            filename (str):                 Name of file containing the words
        """        
        if '.pickle' not in filename:
                filename = filename + '.pickle'
        word2id = pickle.load(open(os.path.join(review_dir, filename)))
        self.create_decoding()

        return word2id


class ReviewDataset(IterableDataset):

    def __init__(self, file, encoding):
        self.file = file
        self.encoding = encoding

    def parse_file(self):
        with open(self.file, 'r') as review_file:
            reader = csv.reader(review_file)
            for line in reader:             
                yield from line

    def get_stream():
        pass

    def __iter__(self):
        return self.parse_file()



def one_hot_embedding(x, embed_size):
    """
    Converts a list of word-indices to their corresponding one-hot respresentation.

    Params:
        x: a torch.LongTensor shaped [seq_len]
        embed_size: size of the one-hot vector. Usually equal to dict_size.
    """
    x_hot = one_hot(x, num_classes=embed_size).float()
    return x_hot

def nnEmbedding(x, embed_size):
    """
    Converts a list of word-indices to an embedding using PyTorch's nn.Embedding() module.

    Params:
        x: a torch.LongTensor shaped [seq_len]
        embed_size: size of the real-valued vector
    """


class Embedder:
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
                raise Exception("If using one-hot, dict_size must equal embed_dim!")
        
        if method == "nnEmbedding":
            self.embedding = nn.Embedding(dict_size, embedding_dim)
            self.embedding.weight.requires_grad=False

    def one_hot_embedding(self, x):
        x_hot = one_hot(x, num_classes=self.embedding_dim)
        return x_hot.float()

    def embed(self, x):
        if self.method == "onehot":
            return self.one_hot_embedding(x)
        elif self.method == "nnEmbedding":
            return self.embedding(x)


class Collator():
    
    def __init__(self, encoding, embedder):
        """
            encoding: word2id dictionary
            embedder: instance of class Embedder
        """
        self.encoding = encoding
        self.dict_size = len(encoding)
        self.embedder = embedder

    def __call__(self, batch):
        
        X = []
        X_len = []
        Y = []
        #Y_len = []

        for line in batch:
            # Represent the line (review) as a list of integers
            encoded_line = [self.encoding[word.lower()] if word.lower() in self.encoding.keys()
            else self.encoding["<UNK>"]
            for word in line.split(' ')]
            encoded_line.insert(0, self.encoding["<SOR>"])
            encoded_line.append(self.encoding["<EOR>"])
            
            # Embed the inputs of the sequence x = [x1, ... xT]
            x = torch.LongTensor(encoded_line[:-1])
            x = embedder.embed(x)

            # The labels y = [y1, ..., yT] don't have to be embedded for CrossEntropyLoss
            y = torch.LongTensor(encoded_line[1:])

            X.append(x)
            X_len.append(len(encoded_line[:-1]))
            Y.append(y)            
        
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        Y_padded = pad_sequence(Y, batch_first=True, padding_value=0)  
        X_packed = pack_padded_sequence(X_padded, X_len, batch_first=True, enforce_sorted=False)
        
        #return X_packed, X_len, Y_padded, Y_len
        return X_packed, Y_padded
        








    