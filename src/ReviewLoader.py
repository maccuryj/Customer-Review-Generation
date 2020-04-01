import os
import csv
import numpy
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

class ProductReviews():    

    def __init__(self, dir):
        """
        Dataset creation class, providing on instantiation a stratified train/test split.
        Provides review dataloader for easy integration with PyTorch.

        Attributes:
            root_dir (str):                 absolute path of review directory
        """
        self.review_dir = review_dir

        reviews = get_reviews()


    def get_reviews(self, size):
        corpus = []
        product2idx = {}        
        for i, file in enumerate(os.listdir(self.dir)):
            if '.csv' in file:
                product2idx[i] = file[:-4]
                with open(os.path.join(data_folder, file), newline='') as f:
                    reader = csv.reader(f)
                    for i, row in enumerate(reader):
                        while len(corpus) < size:
                            corpus.append(row[0])

class ReviewDataset(IterableDataset):

    def __init__(self, file):
        self.file = file

    def parse_file(self):
        with open(self.file, 'r') as review_file:
            reader = csv.reader(review_file)
            for line in reader:
                yield line


if __name__ == "__main__":

    dataset = ReviewDataset("reviews.csv")
    loader = DataLoader(dataset, batch_size=3)

    for inp in loader:
        print(inp)
    