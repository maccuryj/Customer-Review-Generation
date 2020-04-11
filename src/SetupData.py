import numpy as np
import os
import re
import requests
import csv
import json
import gzip
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
import spacy
nlp = spacy.load('en_core_web_sm')

from ReviewLoader import ReviewDataset

from torch.utils.data import IterableDataset
from itertools import chain

class SetupData():
    """
    Class for the initial setup of review data. Assumes that the .json.gz review files were downloaded from 
    "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/".

    Attributes:
        data_folder (str):                  Path of the folder that holds review data
        datasets (str []):                  Array of names of datasets retrieved from the Amazon Review Corpus
        n_train_reviews (int):              Number of training reviews per product dataset
        n_test_reviews (int):               Number of test reviews per product dataset
    """

    def __init__(self, data_folder, datasets, n_train_reviews, n_test_reviews, create_dir=False):
        self.data_folder = data_folder
        self.datasets = datasets
        self.n_train_reviews = n_train_reviews
        self.n_test_reviews = n_test_reviews

        if create_dir is True and not os.path.exists(data_folder):
            os.mkdir(data_folder)

    def _adjust_string(self, s):
        """
        Lowercase, trim, and remove non-letter characters
        https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
        """
        s = s.lower().strip()
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def _filter_review(self, review):
        tokenized_review = nlp(review)
        noun = 0
        verb = 0
        adj = 0
        n_sentences = 0

        for i, tok in enumerate(tokenized_review):
            if tok.pos_ == 'NOUN':
                noun = 1
            if tok.pos_ == 'VERB':
                verb = 1
            if tok.pos_ == 'ADJ':
                adj = 1
            if tok.tag_ == '.':
                n_sentences += 1
            if n_sentences == 2:
                tokenized_review = tokenized_review[:i+1]
                break

        if noun + verb + adj > 1:
            s = ' '.join(tok.text for tok in tokenized_review)
            return self._adjust_string(s)
        else:
            return -1

    def _save_reviews(self, reviews, filename, filetype='csv'):
        if filetype not in ['csv', 'npy']:
            raise ValueError('filetype argument not valid')
        else:
            print("Saving: ", filename)
            if filetype is 'csv':                
                with open(os.path.join(self.data_folder, filename), 'w') as csvfile:
                    writer = csv.writer(csvfile)
                    for i, rev in enumerate(reviews):
                        writer.writerow([rev])
            if filetype is 'npy':
                filename = os.path.join(self.data_folder, filename)
                np.save(filename, embeddings)


    def _reviews_json2csv(self, dataset):
        train_reviews = []
        test_reviews = []
        i = 0
        print("Processing: ", dataset)
        with gzip.GzipFile(dataset + '.json.gz', 'r') as f:
            for line in f.readlines():
                if i == self.n_train_reviews + self.n_test_reviews:
                    break   
                d = json.loads(line)
                try:
                    review = self._filter_review(d["reviewText"])
                except KeyError:
                    review = -1
                if review != -1:
                    if i <= self.n_train_reviews:
                        train_reviews.append(review)
                    else:
                        test_reviews.append(review)
                    i = i + 1

        self._save_reviews(train_reviews, dataset + "_train.csv", 'csv')
        self._save_reviews(test_reviews, dataset + "_test.csv", 'csv')

        return

    def create_csv_files(self):
        for dataset in self.datasets:
            self._reviews_json2csv(dataset)            


    def _reviews2BERT(self, dataset, n_reviews, batch_size, model):
        
        batch_size = batch_size

        print("Processing: ", dataset)
        embeddings = np.empty((n_reviews, 768,))
        review_loader = DataLoader(ReviewDataset([os.path.join(self.data_folder, dataset + '.csv')]), batch_size=batch_size)

        for i, rev in enumerate(review_loader):
            
            if i % round(n_reviews/10) == 0 and i != 0:
                print("Processed: " , n_reviews/i, "%")

            encoding = model.encode(rev)

            for j, enc in enumerate(encoding):
                embeddings[batch_size*i+j] = enc

        self._save_reviews(embeddings, dataset, 'npy')

        return

    def create_embedding_files(self, batch_size):
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        model.to(device)

        train_datasets = [filename + '_train' for filename in self.datasets]
        for train_set in train_datasets:                                    
            self._reviews2BERT(train_set, self.n_train_reviews, batch_size, model)

        test_datasets = [filename + '_test' for filename in self.datasets]
        for test_set in test_datasets:
            self._reviews2BERT(test_set, self.n_test_reviews, batch_size, model)



    def test_indices(self, dataset, i=10, atol=1e-05):
        """
        This function tests whether the indices in the embedding files match those in the
        .csv files. This is important, as those indices are later used for matching reviews
        with their cluster.
        """
        
        reviews = []
        with open(os.path.join(data_folder, dataset + '.csv')) as f:
            reader = csv.reader(f)
            for j, row in enumerate(reader):
                if j < i:
                    reviews.append(row[0])
        
        embeddings = np.load(os.path.join(data_folder, dataset + '_Embedding.npy'))
        emb = embeddings[:i]

        model = SentenceTransformer('bert-base-nli-mean-tokens')
        encoding = model.encode(reviews)            

        # Checks numpy array equality. Due to small variations when saving files and converting,
        # we check with a small fault tolerance of 'atol'.
        if np.allclose(emb.astype(np.float32), np.array(encoding), atol=atol):
            return True
        else:
            return False
