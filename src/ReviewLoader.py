import os
import csv
import numpy
import pickle
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

    def create_vocabulary(self, file, max_features=10000):
        """
        Make use of Term Frequency - Inverse Document Frequency to create a vocabulary
        for use in the RNN. The encoding is adjusted for further needs and a decoding
        dictionary is created.
        """
        vectorizer = TfidfVectorizer(max_features=max_features)
        with open(os.path.join(self.review_dir, file), 'r') as f:
            words = vectorizer.fit_transform(f)
        self.word2id = vectorizer.vocabulary_
        
        self.adjust_encoding()
        self.create_decoding()        

        return vectorizer, words

    def adjust_encoding(self):
        if 0 in self.word2id.values():
            for word in self.word2id:
                if self.word2id[word] == 0:
                    self.word2id[word] == len(self.word2id)

        encoding_size = len(self.word2id)
        if "UNK" not in self.word2id.values():
            self.word2id["UNK"] = encoding_size

    def create_decoding(self):
        for word in self.word2id:
            self.id2word[self.word2id[word]] = word

    def save_vocab(tfidf, words, filename):
        pickle.dump(tfidf, open(os.path.join(review_dir, "tfidf.pickle")))
        pickle.dump(words, open(os.path.join(review_dir, "wordlist.pickle")))

    def load_vocab():
        pass


class ReviewDataset(IterableDataset):

    def __init__(self, file, encoding):
        self.file = file
        self.encoding = encoding

    def parse_file(self):
        with open(self.file, 'r') as review_file:
            reader = csv.reader(review_file)
            for line in reader:             
                yield from line

    def __iter__(self):
        return self.parse_file()

class Collator():
    
    def __init__(self, encoding):
        self.encoding = encoding    

    def __call__(self, batch):
        encoded_batch = []
        batch_lengths = []

        for line in batch:
            encoded_line = [self.encoding[word] if word in self.encoding.keys()
            else self.encoding["UNK"]
            for word in line.split(' ')]
            #Do we need One-Hot?
            encoded_batch.append(one_hot(torch.LongTensor(encoded_line), num_classes=len(self.encoding)-1))
            batch_lengths.append(len(encoded_line))
        
        batch_tensor = pad_sequence(encoded_batch, batch_first=True, padding_value=0)
        padded_batch = pack_padded_sequence(batch_tensor, batch_lengths, batch_first=True, enforce_sorted=False)
        return padded_batch, batch_lengths
        




    