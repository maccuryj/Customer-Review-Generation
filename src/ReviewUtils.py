import os
import torch
from joblib import dump, load
from sentence_transformers import SentenceTransformer

class ReviewUtils():

    def __init__(self, data_folder, resource_folder):
        self.data_folder = data_folder
        self.resource_folder = resource_folder

    def get_reviewloader():
        pass

    def get_BERT(self):
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        model.to(device)

        return model
    
    def load_from_disk(self, folder, filename):
        """
        Load a model from disk.

        Args:
            folder (str):                   name of folder containing file (generally resource_folder)
            filename (str):                 name of file containing the model
        """
        filename = os.path.join(folder, filename)
        result = load(filename)

        return result


class ReviewDataset(IterableDataset):
    """
    This class extends the PyTorch IterableDataset class,
    representing a datastream of reviews from the provided files.
    Depending on the use case, it yields different types of data
    that is retrieved in batches by the PyTorch DataLoader.

    Attributes:
        folder (str):                   folder to retrieve data from
        files (str []):                 files to retrieve data from
        ds_type (str):                  dataset type (simple reviews, review embeddings or generated reviews)
    """

    def __init__(self, folder, files, ds_type):
        if ds_type not in ['rev', 'emb', 'train', 'gen']:
            raise ValueError("Argument 'ds_type' was not recognized")     

        self.folder = folder
        # Allow for single filename string and list of filename strings input
        if isinstance(files, list):
            self.files = files
        else:
            self.files = [files]
        self.ds_type = ds_type

    def parse_file(self, file):
        # Return embeddings from .npy and their respective file
        # Used for KMeans clustering
        if self.ds_type is 'emb':
            embeddings = np.load(os.path.join(self.data_folder, file))
            for emb in embeddings:
                yield file, emb
        
        else:
            with open(os.path.join(self.folder, file), 'r') as review_file:
                reader = csv.reader(review_file)

                # Return reviews from a simple .csv review file
                # Used in BERT embedding creation
                if self.ds_type is 'rev':
                    for i, line in enumerate(reader):                           
                        yield line[0]

                # Return an index, reviews and their respective file
                # Used for LSTM training
                if self.ds_type is 'train':
                    for i, line in enumerate(reader):                           
                        yield i, file, line[0]

                #TODO: Remove this and make the generated review files without clusters
                else:
                    for line in reader:                           
                        yield line

    def get_stream(self):        
        return chain.from_iterable(map(self.parse_file, self.files))

    def __iter__(self):
        return self.get_stream()
