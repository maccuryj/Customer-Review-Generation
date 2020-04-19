import os
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

    def process_gen_reviews(self, gen_reviews):
        """
        Take the generated reviews and split them in a list of clusters and
        token-free reviews that can be clustered.
        """
        clusters = [rev[5] for rev in gen_reviews]
        # Condition on whether there is <EOR> token
        reviews = [rev[8:-6] if rev[-1]=='>' else rev[8:] for rev in gen_reviews]

        return clusters, reviews    

    def predict_gen(self, embedding_model, clustering):
        embedding_model.encode(gen_reviews)
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
        preds = self.predict_gen(embedding_model, clustering)

        acc = accuracy_score(np.array(clusters), np.array(preds))
        conf = confusion_matrix(np.array(clusters), np.array(preds))

        return acc, conf

