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

    def __init__(self, folder=None, datasets=None, n_train_reviews=0, n_test_reviews=0, create_dir=False):
        """
        Assignment of class attributes and creation of review folder in case it does not yet exist

        Args:
            create_dir (bool):              Set True to create directory on first run
        """
        # Make arguments conditional
        self.folder = folder
        self.datasets = datasets
        self.n_train_reviews = n_train_reviews
        self.n_test_reviews = n_test_reviews

        if create_dir is True and not os.path.exists(folder):
            os.mkdir(folder)

    def _clean_review_syntactic(self, s):
        """
        cleans review as string syntactically,
        maps numbers to "<NUM>", maps punctuation to "."

        Args:
        s (str):                       Single review string
        """
        # specific words (capitalisation)
        s = re.sub('(U.S.|USA)', 'United States', s)
        s = re.sub('M(s|rs)\s\.?\s?[a-zA-Z]\w*', 'misses', s)
        s = re.sub('Mr\s\.?\s?[a-zA-Z]\w*', 'mister', s)

        # general
        s = s.lower().strip()

        # specific words (lower)
        s = re.sub('&', 'and', s) 
        s = re.sub("aren't", "are not", s) 
        s = re.sub("can't", "cannot", s) 
        s = re.sub("couldn't", "could not", s) 
        s = re.sub("didn't", "did not", s) 
        s = re.sub("doesn't|doesnt", "does not", s) 
        s = re.sub("don't|dont", "do not", s)
        s = re.sub("hadn't", "had not", s) 
        s = re.sub("hasn't", "has not", s) 
        s = re.sub("haven't", "have not", s) 
        s = re.sub("he'd", "he would", s) 
        s = re.sub("he'll", "he will", s) 
        s = re.sub("he's", "he is", s) 
        s = re.sub("i'd", "i would", s) 
        s = re.sub("i'll", "i will", s) 
        s = re.sub("i'm|im", "i am", s) 
        s = re.sub("i've", "i have", s) 
        s = re.sub("isn't", "is not", s) 
        s = re.sub("it's", "it is", s)
        s = re.sub("let's", "let us", s)  
        s = re.sub("mustn't", "must not", s)  
        s = re.sub("shan't", "shall not", s)  
        s = re.sub("she'd", "she would", s)  
        s = re.sub("she'll", "she will", s)  
        s = re.sub("she's", "she is", s)  
        s = re.sub("shouldn't", "should not", s)  
        s = re.sub("that's", "that is", s)  
        s = re.sub("there's", "there is", s)  
        s = re.sub("they'd", "they would", s)  
        s = re.sub("they'll", "they will", s)
        s = re.sub("they're", "they are", s)  
        s = re.sub("they've", "they have", s)  
        s = re.sub("we'd", "we would", s)  
        s = re.sub("we're", "we are", s)  
        s = re.sub("we've", "we have", s)  
        s = re.sub("weren't", "were not", s)  
        s = re.sub("what'll", "what will", s)  
        s = re.sub("what're", "what are", s) 
        s = re.sub("what's", "what is", s) 
        s = re.sub("what've	", "what have", s) 
        s = re.sub("where's", "	where is", s) 
        s = re.sub("who'd", "who would", s) 
        s = re.sub("who'll", "who will", s) 
        s = re.sub("who're", "who are", s) 
        s = re.sub("who's", "who is", s) 
        s = re.sub("who've", "who have", s) 
        s = re.sub("won't", "will not", s) 
        s = re.sub("wouldn't", "would not", s) 
        s = re.sub("you'd", "you would", s) 
        s = re.sub("you'll", "you will", s) 
        s = re.sub("you're", "you are", s) 
        s = re.sub("you've", "you have", s)

        # punctuation: map 
        s = re.sub('\.+|!+|\?+', '.', s)

        # punctuation: delete
        s = re.sub(r"[^0-9a-zA-Z<>\.]+", " ", s)

        # numbers
        #s = re.sub('\d+([a-z]+)|([a-z])+\d+|\d+|[a-z]+\d+[a-z]+', '<NUM>', s)
        return s

    def _filter_review(self, review):
        """
        Part of initial review preprocessing. The reviews are filtered and only those are kept that
        have 2 of the three: Noun, Verb, Adjective.
        Reviews are trimmed to two phrases.

        Args:
        review (str):                       Single review string
        """
        #review  = self._clean_review_syntactic(review) 
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
            return s
        else:
            return -1

    def _save_reviews(self, reviews, filename, filetype='csv'):
        """
        Helper function to save reviews both as .csv and .npy.
        
        Args:
            reviews (str []):               Set of reviews to be stored on disk
            filename (str):                 Review set filename, in our example the product set and train/test
            filetype (str):                 Type of file to save (either .csv or .npy)
        """
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
                np.save(filename, reviews)


    def _reviews_json2csv(self, dataset):
        """
        Builds the review .csv files from the .json.gz zipped files from the online Amazon Review corpus.
        Preprocessing and filtering of reviews takes place in this function.

        Args:
            dataset (str):                  Name (product) of dataset to be processed
        """
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
                    if i < self.n_train_reviews:
                        train_reviews.append(review)
                    else:
                        test_reviews.append(review)
                    i = i + 1

        self._save_reviews(train_reviews, dataset + "_train.csv", 'csv')
        self._save_reviews(test_reviews, dataset + "_test.csv", 'csv')

        return

    def create_csv_files(self):
        """
        Runs the .csv file creation processes on the set of supplied datasets
        """
        for dataset in self.datasets:
            self._reviews_json2csv(dataset)            


    def reviews2BERT(self, loader, n_reviews, batch_size, model, filename=None, save_embeddings=False):
        """
        Store BERT Embeddings built from the reviews taken from the .csv files
        These embeddings are used for clustering purposes
        A DataLoader is used to retrieve the reviews from the .csv files to avoid memory overload

        Args:
            dataset (str):                      Product name of dataset
            n_reviews (int):                    Number of reviews that are processed in the given dataset
            batch_size (int):                   Size of batches loaded from the DataLoader
            model (sentence_transformer):       BERT Embedding encoder model
        """
        batch_size = batch_size

        print("Processing: ", dataset)
        embeddings = np.empty((n_reviews, 768,))        

        for i, rev in enumerate(loader):
            
            if i % round(n_reviews/10) == 0 and i != 0:
                print("Processed: " , n_reviews/i, "%")

            encoding = model.encode(rev)

            for j, enc in enumerate(encoding):
                embeddings[batch_size*i+j] = enc

        if save_embeddings and filename is not None:
            self._save_reviews(embeddings, filename, 'npy')

        return embeddings

    def get_BERT(self):
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        model.to(device)

        return model

    def get_reviewloader(self, folder, files, batch_size):        
        dataset = ReviewDataset(folder, files + '.csv', 'rev')
        loader = DataLoader(dataset, batch_size=batch_size)

        return loader

    def create_embedding_files(self, batch_size):
        """
        Runs the BERT embedding file creation processes on the set of supplied datasets

        Args:
            batch_size (int):                   Size of batches to be loaded from the DataLoader
        """
        model = get_BERT()

        train_datasets = [filename + '_train' for filename in self.datasets]
        for train_set in train_datasets: 
            loader = get_review_loader(self.folder, [train_set], batch_size)                                   
            self._reviews2BERT(loader, self.n_train_reviews, batch_size, model, dataset, True)

        test_datasets = [filename + '_test' for filename in self.datasets]
        for test_set in test_datasets:
            loader = get_review_loader(self.folder, [test_set], batch_size) 
            self._reviews2BERT(loader, self.n_test_reviews, batch_size, model, dataset, True)



    def test_indices(self, dataset, i=10, atol=1e-05):
        """
        This function tests whether the indices in the embedding files match those in the
        .csv files. This is important, as those indices are later used for matching reviews
        with their cluster.

        Args:
            dataset (str):                      Product name of dataset
            i (int):                            Index of review until which comparison is made
            atol (float):                       Very small floating point number, supplied as fault tolerance for value comparison
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
