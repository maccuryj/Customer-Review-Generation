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

from src.ReviewLoader import ReviewDataset

class SetupData():
    """
    Class for the initial setup of review data. Assumes that the .json.gz review files were downloaded from 
    "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/".

    Attributes:
        data_folder (str):                  Path of the folder that holds review data
        datasets (str []):                  Array of names of datasets retrieved from the Amazon Review Corpus
        n_reviews (int):                    Number of reviews per dataset
    """

    def __init__(self, data_folder, datasets, create_dir=False):
        self.data_folder = data_folder
        self.datasets = datasets

        if create_dir is True and not os.path.exists(data_folder):
            os.mkdir(data_folder)

    def _adjust_string(s):
        """
        Lowercase, trim, and remove non-letter characters
        https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
        """
        s = s.lower().strip()
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def filter_review(self, review):
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
            return _adjust_string(s)
        else:
            return -1

    def reviews_json2csv(self, dataset):
        reviews = []
        i = 0
        print("Processing: ", dataset)
        with gzip.GzipFile(dataset + '.json.gz', 'r') as f:
            for line in f.readlines():
                if i == self.n_reviews:
                    break   
                d = json.loads(line)
                try:
                    review = filter_review(d["reviewText"])
                except KeyError:
                    review = -1
                if review != -1:
                    reviews.append(review)
                    i = i+1    
        print("Saving " + dataset + ".csv")
        with open(os.path.join(self.data_folder, dataset + '.csv'), 'w') as csvfile:
            writer = csv.writer(csvfile)
            for i, rev in enumerate(reviews):
                writer.writerow([rev])

        return

    def create_csv_files(self, n_reviews):
        self.n_reviews = n_reviews
        for dataset in self.datasets:
            self.reviews_json2csv(dataset)            


    def reviews2BERT(self, dataset, batch_size, num_workers):
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        model.to(device)

        batch_size = batch_size
        num_workers = num_workers

        print("Processing: ", dataset)
        embeddings = np.empty((self.n_reviews, 768,))
        review_loader = DataLoader(ReviewDataset([os.path.join(self.data_folder, dataset + '.csv')]), batch_size=batch_size, num_workers=num_workers)

        for i, rev in review_loader:
            
            if i % 5000 == 0 and i != 0:
                print("Processed: " , self.n_reviews/i, "%")

            encoding = model.encode(rev)
            for j, enc in enumerate(encoding):
                embeddings[batch_size*i+j] = enc

        print("Saving " + dataset + "_Embedding.npy")
        filename = os.path.join(self.data_folder, dataset + "_Embedding")
        np.save(filename, embeddings)

        return

        def create_embedding_files(self, batch_size, num_workers):
            for dataset in datasets:                                    
                self.reviews2BERT(dataset, batch_size, num_workers)


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
