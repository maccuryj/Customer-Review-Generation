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

from ReviewUtils import ReviewUtils
from ReviewLoader import ReviewDataset

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

    def __init__(self, utils, files=None):
        """
        Assignment of class attributes and creation of resource folder to store model 
        and dictionary, in case it does not yet exist
        """
        if not isinstance(utils, ReviewUtils):        
            raise ValueError("Argument 'utils' should be a ReviewUtils object!")
        self.utils = utils
        self.files = files

    #can be moved to utils
    def get_loader(self, batch_size):
        """
        Returns a DataLoader for review embeddings, on the basis of
        a set of files.

        Args:
            batch_size (int):               batch size for embedding DataLoader
            folder (str):                   name of folder that holds files
            files (str []):                 list of embedding filenames
        """
        dataset = ReviewDataset(self.utils.folder, self.utils.files, 'emb')
        self.loader = DataLoader(dataset, batch_size=batch_size)

        return self.loader

    def compute_clusters(self, loader, clustering, save_labels=False, filename='ClusterDict.joblib'):
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
            self.utils.save_to_disk(self.utils.resource_folder, filename, cluster_dict)            

        return cluster_dict      

    #TODO: loader has to be passed or separate to have a computation function
    def MB_Spherical_KMeans(self, k, batch_size=2048, save_model=True, save_labels=True, fn_model='KMeansModel.joblib', fn_labels='ClusterDict.joblib'):
        """
        MiniBatchKMeans model, clustering the normalized BERT Embeddings.
        
        Args:
            k (int):                        Number of resulting clusters
            batch_size (int):               Size of batches returned by the dataloader
            save_model (bool):              Decides whether model should be saved
            save_labels (bool):             Decides whether cluster labels should be saved
        """
        loader = self.get_loader(batch_size)        
        clustering = MiniBatchKMeans(n_clusters=k, batch_size=batch_size)

        for f, batch in loader:
            normalize(batch)
            clustering.partial_fit(batch)
        
        if save_model:
            self.utils.save_to_disk(self.utils.resource_folder, fn_model, clustering)

        cluster_dict = self._compute_clusters(loader, clustering, save_labels, fn_labels)

        return clustering, cluster_dict

    #TODO: Loader has to be passed
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
        loader = self.get_loader(batch_size) 

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
        embeddings = np.load(os.path.join(self.utils.data_folder, file))
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

        

