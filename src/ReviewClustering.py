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
    """
    Dataset class that serves as basis for the BERT Embedding Dataloader
    used for clustering purposes.
    Implemented as IterableDataset in order to iteratively retrieve batches from
    disk and prevent memory overload.
    """

    def __init__(self, data_folder, files):
        self.files = files
        self.data_folder = data_folder

    def parse_file(self, file):
        """
        Reads from the .npy embedding files and yields a file vector
        together with the embeddings
        """
        embeddings = np.load(os.path.join(self.data_folder, file))
        for i, emb in enumerate(embeddings):
            yield file, emb

    def get_stream(self):   
        """
        Implementation of stream over provided embedding files
        """     
        return chain.from_iterable(map(self.parse_file, self.files))

    def __iter__(self):
        return self.get_stream()


class ReviewKMeans():
    """
    Clustering process with MiniBatchKMeans
    Includes methods to find optimal K,
    save or load models and ultimately store cluster labels

    Attributes:
        data_folder (str):                  filename of review data folder
        resource_folder (str):              filename of resource folder
        files (str []):                     set of review dataset filenames
        loader (DataLoader):                DataLoader for review embeddings
        model (MiniBatchKMeans):            MiniBatchKMeans clustering model
        cluster_dict (dict):                dictionary of cluster labels
    """

    def __init__(self, data_folder, resource_folder, files):
        """
        Assignment of class attributes and creation of resource folder to store model 
        and dictionary, in case it does not yet exist
        """
        self.files = files
        self.data_folder = data_folder
        self.resource_folder = resource_folder
        if not os.path.exists(resource_folder):
            os.mkdir(resource_folder)

    def get_embeddingloader(self, batch_size):
        """
        Returns a DataLoader for review embeddings

        Args:
            batch_size (int):               batch size for embedding DataLoader
        """
        dataset = ReviewEmbeddingDataset(self.data_folder, self.files)
        self.loader = DataLoader(dataset, batch_size=batch_size)

        return self.loader

    def load_model(self, folder=None, filename='KMeansModel.joblib'):
        """
        Load a model from disk.

        Args:
            folder (str):                   name of folder containing file (generally resource_folder)
            filename (str):                 name of file containing the model
        """        
        if folder is None:
            folder = self.resource_folder
        filename = os.path.join(folder, filename)
        self.model = load(filename)
        return self.model

    def load_labels(self, folder=None, filename='ClusterDict.joblib'):
        """
        Load cluster labels from disk.

        Args:
            folder (str):                   name of folder containing file (generally resource_folder)
            filename (str):                 name of file containing the cluster labels
        """        
        if folder is None:
            folder = self.resource_folder
        filename = os.path.join(folder, filename)
        self.cluster_dict = load(filename)
        return self.cluster_dict

    def compute_clusters(self, loader, clustering, save_labels=False, filename="ClusterDict.joblib"):
        """
        Computes the cluster labels given a KMeans model and stores them on disk

        Args:
            loader (DataLoader):            DataLoader for review embeddings
            clustering (model):             KMeans model
            save_labels (bool):             Decides whether cluster labels should be saved

        """
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
            dump(self.cluster_dict, os.path.join(self.resource_folder, filename))

        return self.cluster_dict      


    def MB_Spherical_KMeans(self, k, batch_size=2048, save_model=True, save_labels=True):
        """
        MiniBatchKMeans model, clustering the normalized BERT Embeddings.
        
        Args:
            k (int):                        Number of resulting clusters
            batch_size (int):               Size of batches returned by the dataloader
            save_model (bool):              Decides whether model should be saved
            save_labels (bool):             Decides whether cluster labels should be saved
        """
        loader = self._get_embeddingloader(batch_size)        
        clustering = MiniBatchKMeans(n_clusters=k, batch_size=batch_size)

        for f, batch in loader:
            normalize(batch)
            clustering.partial_fit(batch)

        self.model = clustering
        if save_model:
            dump(clustering, os.path.join(self.resource_folder, "KMeansModel.joblib"))

        cluster_dict = self._compute_clusters(loader, clustering, save_labels)

        return self.model, cluster_dict


    def elbow_plot(self, min_k=20, max_k=300, step=20, notification_step=100, batch_size=2048):
        """
        Creates an elbow plot for the KMeans clustering
        based on the provided K values
        and the resulting within-cluster SSQ

        Args:
            min_k (int):
            max_k (int):
            step (int):
            notification_step (int):
            batch_size (int):
        """
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
        embeddings = np.load(os.path.join(self.data_folder, file))
        emb = embeddings[:i]
        del(embeddings)
        preds = self.model.predict(emb)
        
        labels = []
        for j in range(1, i+1):
            labels.append(self.cluster_dict[file + ' - ' + str(j)])
        
        if np.array_equal(preds, np.array(labels)):
            return True
        else:
            return False

        

