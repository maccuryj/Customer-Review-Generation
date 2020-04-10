import numpy as np
import os
import csv
from joblib import dump, load
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})

from src.ReviewLoader import ReviewDataset

class ReviewClustering():

    def __init__(review_dir):
         self.review_dir = review_dir 

    def get_reviews(datasets=None):
        reviews = []
        review_files = [file for file in os.listdir(self.review_dir) if '.csv' in file]
        for file in review_files:
            with open(os.path.join(self.review_dir, file), newline='') as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    reviews.append(row[1])

        return reviews       

    def get_embeddings(n_files, n_reviews, review_vector_files=None):        
        embeddings = np.empty((n_files, n_reviews, 768))
        if review_vector_files is None:
            review_vector_files = [file for file in os.listdir(self.review_dir) if '.npy' in file]        
        
        for i, file in enumerate(review_vector_files):
            embeddings[i] = np.load(os.path.join(self.review_dir, file))

        self.embeddings = embeddings 
        return embeddings    

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

    def parse_file(self, file):
        embeddings = np.load(os.path.join(data_folder, file + '_Embedding.npy'))
        for i, emb in enumerate(embeddings):
            yield file, emb

    def get_stream(self):        
        return chain.from_iterable(map(self.parse_file, self.files))

    def __iter__(self):
        return self.get_stream()

class ReviewKMeans():

    def __init__(self, files):
        self.files = files

    def _get_embeddingloader(self, batch_size):
        dataset = ReviewEmbeddingDataset(self.files)
        self.loader = DataLoader(dataset, batch_size=batch_size)

        return self.loader

    def _compute_clusters(self, loader, clustering):
        self.cluster_dict = {}

        for i, batch in enumerate(loader):
            print(batch)


    def MB_Spherical_KMeans(self, k, batch_size=2048, save=True):

        loader = get_embeddingloader()        
        clustering = MiniBatchKMeans(n_clusters=k, batch_size=batch_size)
        normalizer = Normalizer(copy=False)

        for batch in loader:
            print(batch)
            #Preserve file processing order, so indices can be matched
            if f != file:
                file = f
                files.append(file)

            normalize(batch)
            clustering.partial_fit(batch)

        if save:
            dump(clustering, "KMeansModel.joblib")

        self._compute_clusters(loader, clustering)



    def elbow_plot(min_k=20, max_k=300, step=20, notice_step=100, batch_size=2048):
        ssq = []

        for k in range(min_k, max_k, step):
            if k % 100 == 0:
                print("K: ", k)
            clustering = MiniBatchKMeans(n_clusters=k, batch_size=batch_size)
            normalizer = Normalizer(copy=False)
            spherical_kmeans = make_pipeline(normalizer, clustering)

            spherical_kmeans.fit(samples)

            ssq.append(spherical_kmeans.inertia_)

        sns.lineplot(np.arange(min_k,max_k,step), ssq)
