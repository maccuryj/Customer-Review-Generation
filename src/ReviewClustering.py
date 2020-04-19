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

    Args:
        utils (ReviewUtils):                Utility object that holds information and helper classes
    """

    def __init__(self, utils):
         if not isinstance(utils, ReviewUtils):        
            raise ValueError("Argument 'utils' should be a ReviewUtils object!")
        self.utils = utils

    def get_reviews(self, datasets=None):
        reviews = []
        review_files = [file for file in os.listdir(self.utils.data_folder) if '.csv' in file]
        for file in review_files:
            with open(os.path.join(self.utils.data_folder, file), newline='') as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    reviews.append(row[1])

        return reviews       

    def get_embeddings(self, n_files, n_reviews, review_vector_files=None):        
        embeddings = np.empty((n_files, n_reviews, 768))
        if review_vector_files is None:
            review_vector_files = [file for file in os.listdir(self.utils.data_folder) if '.npy' in file]        
        
        for i, file in enumerate(review_vector_files):
            embeddings[i] = np.load(os.path.join(self.utils.data_folder, file))

        self.embeddings = embeddings 
        return embeddings    

    def sample_reviews(self, ratio=0.1):
        sample = []
        indices = []
        for i, product in enumerate(self.embeddings):
            for j, rev in enumerate(product):
                if np.random.sample() < ratio:
                    indices.append(i*self.embeddings.shape[1] + j)
                    sample.append(rev)

        return samples

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

    def __init__(self, data_folder=None, resource_folder=None, files=None):
        """
        Assignment of class attributes and creation of resource folder to store model 
        and dictionary, in case it does not yet exist
        """
        #Make files argument optional
        self.files = files
        self.data_folder = data_folder
        self.resource_folder = resource_folder
        if resource_folder is not None and not os.path.exists(resource_folder):
            os.mkdir(resource_folder)

    def get_embedding_file_loader(self, batch_size, folder=None, files=None):
        """
        Returns a DataLoader for review embeddings, on the basis of
        a set of files.

        Args:
            batch_size (int):               batch size for embedding DataLoader
            folder (str):                   name of folder that holds files
            files (str []):                 list of embedding filenames
        """
        if folder is None and self.data_folder is not None:
            folder = self.data_folder
        if files is None and self.files is not None:
            files = self.files
            
        dataset = ReviewDataset(folder, files, 'emb')
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
            filename (str):                 Name of cluster dictionary file to be stored
        """
        cluster_dict = {}

        i = 1
        curr_file = ""
        for files, batch in loader:
            preds = clustering.predict(batch)
            for j in range(len(files)):
                print(i)
                #Reset index when changing file
                if files[j] != curr_file:
                    curr_file = files[j]
                    i = 1             
                cluster_dict[files[j] + ' - ' + str(i)] = preds[j]
                i = i + 1

        if save_labels:
            dump(cluster_dict, os.path.join(self.resource_folder, filename))

        return cluster_dict      


    def MB_Spherical_KMeans(self, k, batch_size=2048, save_model=True, save_labels=True):
        """
        MiniBatchKMeans model, clustering the normalized BERT Embeddings.
        
        Args:
            k (int):                        Number of resulting clusters
            batch_size (int):               Size of batches returned by the dataloader
            save_model (bool):              Decides whether model should be saved
            save_labels (bool):             Decides whether cluster labels should be saved
        """
        loader = self._get_embedding_file_loader(batch_size)        
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

        

