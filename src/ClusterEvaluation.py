import os
import csv
import torch
import numpy as np
import seaborn as sns
from ReviewUtils import ReviewUtils
from sklearn.metrics import accuracy_score, confusion_matrix

class ClusterEvaluation():

    def __init__(self, utils):
        if not isinstance(utils, ReviewUtils):        
            raise ValueError("Argument 'utils' should be a ReviewUtils object!")
        self.utils = utils

    def cluster_samples(self, cluster_dict, sample_size=10, clusters=None):
        sample_files = []
        indices = []
        v_clusters = []
        cluster_samples = {}

        if clusters is None:
            clusters = set(cluster_dict.values())

        for c in clusters:
            cluster_samples[c] = []
        
        i = 1
        for f in range(5):            
            for k in list(cluster_dict.keys())[f:(f+1)*100000]:
                if i % ((sample_size*len(clusters))/5)+1:
                    i += 1
                    break
            if cluster_dict[k] in clusters:
                if len(cluster_samples[cluster_dict[k]]) < sample_size:
                    #cluster_samples[cluster_dict[k]].append(k)
                    sample_files.append(k[:c])
                    indices.append(k[c+7:])
                    v_clusters.append(cluster_dict[k])
                    i += 1

        j = 0
        for fn in set(sample_files):            
            with open(fn, 'r') as f:
                reader = csv.reader(f)
            
            for i, row in enumerate(reader):
                if sample_files[j] != fn:
                    break
                if i == indices[j]:
                    cluster_samples[v_clusters[j]].append(row)

        return cluster_samples

        

        """                        
        for j, k in enumerate(cluster_dict.keys()):
            if i % ((sample_size*len(clusters))/5):

            if cluster_dict[k] in clusters:
                if cluster_dict[k] not in cluster_samples.keys():
                    cluster_samples[cluster_dict[k]] = [k]
                else:
                    if len(cluster_samples[cluster_dict[k]]) < sample size:
                        cluster_samples[cluster_dict[k]].append(k)
                        i += 1
        """
        
        



    def process_gen_reviews(self, gen_reviews):
        """
        Take the generated reviews and split them in a list of clusters and
        token-free reviews that can be clustered.
        """
        clusters = []
        reviews = []
        for rev in gen_reviews:
            label_delimiter = rev.find('>')
            clusters.append(int(rev[5:label_delimiter]))
            if rev[-1]=='>':
                reviews.append(rev[label_delimiter+2:-6])
            else:
                reviews.append(rev[label_delimiter+2:])

        return clusters, reviews    

    def predict_gen(self, gen_reviews, embedding_model, clustering):
        embeddings = embedding_model.encode(gen_reviews)
        preds = clustering.predict(embeddings)

        return preds

    def plot_conf(self, conf):
        sns.heatmap(conf, annot=True).set_title("Confusion Matrix for conditionally generated reviews")

    def eval_clustering(self, gen_reviews, clusters, fn_clustering='KMeansModel.joblib'):
        bert_model = self.utils.get_BERT()
        clustering = self.utils.load_from_disk(self.utils.resource_folder, fn_clustering)
        acc, conf = self._eval_clustering(gen_reviews, clusters, bert_model, clustering)
        
        return acc, conf

    def _eval_clustering(self, gen_reviews, clusters, embedding_model, clustering):
        """
        Separate computation class, in case different models should be used
        """
        result = []
        preds = self.predict_gen(gen_reviews, embedding_model, clustering)

        acc = accuracy_score(np.array(clusters), np.array(preds))
        conf = confusion_matrix(np.array(clusters), np.array(preds))

        return acc, conf

