import numpy as np
import os
import csv
from joblib import dump, load
from itertools import chain
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, IterableDataset
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})

from ReviewLoader import ReviewDataset

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

    def __init__(self, data_folder, files):
        self.files = files
        self.data_folder = data_folder

    def parse_file(self, file):
        embeddings = np.load(os.path.join(self.data_folder, file))
        for i, emb in enumerate(embeddings):
            yield file, emb

    def get_stream(self):        
        return chain.from_iterable(map(self.parse_file, self.files))

    def __iter__(self):
        return self.get_stream()

class ReviewKMeans():

    def __init__(self, data_folder, files):
        self.files = files
        self.data_folder = data_folder

    def _get_embeddingloader(self, batch_size):
        dataset = ReviewEmbeddingDataset(self.data_folder, self.files)
        self.loader = DataLoader(dataset, batch_size=batch_size)

        return self.loader

    def _compute_clusters(self, loader, clustering, batch_size):
        self.cluster_dict = {}

        i = 0
        curr_file = ""
        for file, batch in loader:
            preds = clustering.predict(batch)
            print(file)
            for j in range(batch_size):
                print(j)                
                print(file[j])
                #Reset index when changing file
                if file[j] != curr_file:
                    curr_file = file[j]
                    i = 0                
                self.cluster_dict[file[j] + ' - ' + str(i)] = preds[j]
                i = i + 1

        print(cluster_dict)


    def MB_Spherical_KMeans(self, k, batch_size=2048, save=True):

        loader = self._get_embeddingloader(batch_size)        
        clustering = MiniBatchKMeans(n_clusters=k, batch_size=batch_size)

        for f, batch in loader:
            normalize(batch)
            clustering.partial_fit(batch)

        if save:
            dump(clustering, "KMeansModel.joblib")

        self._compute_clusters(loader, clustering, batch_size)



    def elbow_plot(self, min_k=20, max_k=300, step=20, notification_step=100, batch_size=2048):
        ssq = []
        n_steps = (max_k-min_k)/step
        loader = self._get_embeddingloader(batch_size) 

        for k in range(min_k, max_k, step):
            if k % notification_step == 0:
                print("K: ", k)            
            clustering = MiniBatchKMeans(n_clusters=k, batch_size=batch_size)
            for f, batch in loader:
                normalize(batch)
                clustering.partial_fit(batch)


            ssq.append(clustering.inertia_)

        sns.lineplot(np.arange(min_k,max_k,step), ssq)
