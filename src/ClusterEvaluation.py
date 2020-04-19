import os
import torch
from ReviewUtils import ReviewUtils

class ClusterEvaluation():

    def __init__(self, utils):
        if not isinstance(utils, ReviewUtils):        
            raise ValueError("Argument 'utils' should be a ReviewUtils object!")
        self.utils = utils

    def process_gen_reviews(self, gen_reviews):
        #TODO: Split into cluster and actual gen review and return those
        pass

    def eval_clustering(self, gen_reviews, clusters, fn_clustering='KMeansModel.joblib'):
        bert_model = self.utils.get_BERT()
        clustering = self.utils.load_from_disk(self.utils.resource_folder, fn_clustering)
        result = _eval_clustering(gen_reviews, clusters, bert_model, clustering)
        
        return result

    def _eval_clustering(self, gen_reviews, clusters, embedding_model, clustering):
        """
        Separate computation class, in case different models should be used
        """
        result = []
        embedding_model.embed(gen_reviews)
        preds = clustering.predict(embeddings)

        for i in range(len(preds)):
            if preds[i] == clusters[i]:
                result.append(1)
            else:
                result.append(0)

    def test_eval(self, fn_gen_reviews, fn_labels):
        gen_reviews = self.utils.load_from_disk(self.utils.data_folder, )
