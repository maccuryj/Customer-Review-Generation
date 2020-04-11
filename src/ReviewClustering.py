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
    """
    Class created mostly to test clustering with a subset of the data.
    Provides helper function for retrieval and sampling of review data.
    """

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

# maybe save model, dict, etc.. in other folder
class ReviewKMeans():

    def __init__(self, data_folder, files):
        self.files = files
        self.data_folder = data_folder

    def _get_embeddingloader(self, batch_size):
        dataset = ReviewEmbeddingDataset(self.data_folder, self.files)
        self.loader = DataLoader(dataset, batch_size=batch_size)

        return self.loader

    def load_model(self, filename='KMeansModel.joblib'):
        self.model = load(filename)
        return self.model

    def load_labels(self, filename='ClusterDict.joblib'):
        self.cluster_dict = load(filename)
        return self.cluster_dict

    def _compute_clusters(self, loader, clustering, save_labels):
        self.cluster_dict = {}

        i = 1
        curr_file = ""
        for files, batch in loader:
            preds = clustering.predict(batch)
            for j in range(len(files)):
                #Reset index when changing file
                if files[j] != curr_file:
                    curr_file = files[j]
                    i = 1             
                self.cluster_dict[files[j] + ' - ' + str(i)] = preds[j]
                i = i + 1

        if save_labels:
            dump(self.cluster_dict, "ClusterDict.joblib")

        return self.cluster_dict      


    def MB_Spherical_KMeans(self, k, batch_size=2048, save_model=True, save_labels=True):

        loader = self._get_embeddingloader(batch_size)        
        clustering = MiniBatchKMeans(n_clusters=k, batch_size=batch_size)

        for f, batch in loader:
            normalize(batch)
            clustering.partial_fit(batch)

        self.model = clustering
        if save_model:
            dump(clustering, "KMeansModel.joblib")

        cluster_dict = self._compute_clusters(loader, clustering, save_labels)

        return self.model, cluster_dict


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


        sns.lineplot(np.arange(min_k,max_k,step), ssq).set_title("KMeans Inertia Elbow Plot")

    def test_indices(self, file, i):
        embeddings = np.load(file)
        emb = embeddings[1, i+1]
        del(embeddings)
        
        preds = self.model.predict(emb)
        print(pred)
        labels = []
        for i in range(1, i+1):
            labels.append(self.cluster_dict[file + ' - ' + i])
        print(labels)
        if pred == labels:
            return True
        else:
            return False

        

