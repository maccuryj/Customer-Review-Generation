import numpy as np
import os
import csv
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import make_pipeline
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})

from src.ReviewLoader import ReviewDataset

class ReviewClustering():

    def __init__(review_dir):
         self.review_dir = review_dir         

    def get_embeddings(shape=(5,50000,768), review_vector_files=None):
        embeddings = np.empty(shape)
        if review_vector_files is None:
            review_vector_files = [file for file in os.listdir(self.review_dir) if '.npy' in file]        
        
        for i, file in enumerate(review_vector_files):
            embeddings[i] = np.load(os.path.join(self.review_dir, file))

        self.embeddings = embeddings    
        return embeddings

    def get_reviews(datasets=None):
        reviews = []
        review_files = [file for file in os.listdir(self.review_dir) if '.csv' in file]
        for file in preview_files:
            with open(os.path.join(self.review_dir, file), newline='') as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    corpus.append(row[0])

        self.reviews = reviews
        return reviews

    def sample_reviews(ratio=0.1):
        sample = []
        indices = []
        for i, product in enumerate(self.embeddings):
            for j, rev in enumerate(product):
                if np.random.sample() < ratio:
                    indices.append(i*self.embeddings.shape[1] + j)
                    sample.append(rev)

        return samples

class ReviewEmbeddingDataset(IterableDataset):

    def __init__(self, files):
        self.files = files

    #np.load
    def parse_file(self, file):
        with open(file, 'r') as review_file:
            reader = csv.reader(review_file)
            for line in reader:             
                yield from line

    def get_stream(self):        
        return chain.from_iterable(map(self.parse_file, self.files))

    def __iter__(self):
        return self.get_stream()

class ReviewKMeans():

    def __init__(embedding_files, batch_size=2048):
        self.embedding_files = embedding_files

    def get_embeddingloader(self):
        dataset = ReviewDataset(self.embedding_files)
        self.loader = DataLoader(dataset, batch_size=self.batch_size)

        return self.loader

    def MB_Spherical_KMeans(self, k):
        clustering = MiniBatchKMeans(n_clusters=k, batch_size=batch_size)
        normalizer = Normalizer(copy=False)
        spherical_kmeans = make_pipeline(normalizer, clustering)

        spherical_kmeans.fit(samples)


    def elbow_plot(min_k=20, max_k=300, step=20):
        ssq = []
        batch_size = 2500

        for k in range(20, 300, 20):
            if k % 20 == 0:
                print(k)
            clustering = MiniBatchKMeans(n_clusters=k, batch_size=batch_size)
            normalizer = Normalizer(copy=False)
            spherical_kmeans = make_pipeline(normalizer, clustering)

            spherical_kmeans.fit(samples)

            ssq.append(spherical_kmeans.inertia_)

        sns.lineplot(np.arange(min_k,max_k,step), ssq)

if __name__ == "__main__":
    data_files = ["reviews.csv", "reviews_2.csv"]
    dataset = ReviewDataset(data_files)
    loader = DataLoader(dataset, batch_size=3)

    for inp in loader:
        print(inp)