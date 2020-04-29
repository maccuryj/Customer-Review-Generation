import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F
from matplotlib import pyplot as plt
import re
import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model:
    """
    A wrapper around a Pytorch-LSTM model for convenient training, evaluating, 
    saving and loading the model.
    """

    def __init__(self, model, h_learnable, optimizer, criterion, embedder):
        self.model = model
        self.h_learnable = h_learnable
        self.optimizer = optimizer
        self.criterion = criterion
        self.embedder = embedder

        self.train_loader = None
        self.test_loader = None

        self.train_losses = []
        self.test_losses = []
        self.reviews = []
        self.cluster_accs = []
        self.cluster_confs = []

    def save_model(self, path):
        if self.train_loader is not None:
            train_files = self.train_loader.dataset.files
            batch_size = self.train_loader.batch_size
        else:
            batch_size = None
            train_files = []
        if self.test_loader is not None:
            test_files = self.test_loader.dataset.files
        else:
            test_files = [] 
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'other_params': {
                "train_files": train_files,
                "test_files": test_files,
                "batch_size": batch_size,
                "embed_method": self.embedder.method,

                "input_size": self.embedder.embedding_dim,
                "output_size": self.embedder.num_embeddings, # = dict_size
                "hidden_size": self.model.hidden_size,
                "num_layers": self.model.num_layers,
                "dropout_prob": self.model.dropout.p,

                "train_losses": self.train_losses,
                "test_losses": self.test_losses,
                "cluster_accs": self.cluster_accs,
                "cluster_confs": self.cluster_confs,
                "reviews": self.reviews
            }
        }, path)

    def _test_epoch(self):
        self.model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for inp in self.test_loader:
                if inp == -1:
                    continue
                X_padded, X_len, Y_padded = inp
                X_packed = pack_padded_sequence(X_padded, X_len, batch_first=True, enforce_sorted=False)
                X_packed = X_packed.to(device=device)
                Y_padded = Y_padded.to(device=device)
                batch_size = Y_padded.shape[0]
                Y_padded = Y_padded.contiguous().view(-1)

                if self.h_learnable:
                    out, h = self.model.predict(X_packed, h=None, batch_size=batch_size)
                else:
                    h = self.model.init_hidden(batch_size)
                    out, h = self.model(X_packed, h, dropout=False)

                loss = self.criterion(out, Y_padded)
                epoch_loss += loss.item()
        return epoch_loss

    def _train_epoch(self):
        self.model.train()
        epoch_loss = 0

        for inp in self.train_loader:
            if inp == -1:
                continue
        
            X_padded, X_len, Y_padded = inp
            X_packed = pack_padded_sequence(X_padded, X_len, batch_first=True, enforce_sorted=False)
            X_packed = X_packed.to(device=device)
            Y_padded = Y_padded.to(device=device) # shape (N, M)
            batch_size = Y_padded.shape[0]
            Y_padded = Y_padded.contiguous().view(-1) # shape (N*M)

            # Flush gradients
            self.model.zero_grad()

            if self.h_learnable:
                out, h = self.model(X_packed, batch_size=batch_size) # out: shape (N*M, output_size)
            else:
                h = self.model.init_hidden(batch_size)
                out, h = self.model(X_packed, h, dropout=True)
            
            loss = self.criterion(out, Y_padded)
            loss.backward()
            epoch_loss += loss.item() # .item() doesn't store the computationalgraph

            # Clip gradients to prevent vanishing/exploding gradients
            nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
        return epoch_loss

    def train(self, num_epochs, train_loader, test_loader, word2id, id2word, clusterEval, n_reviews=1):
        self.train_loader = train_loader
        self.test_loader = test_loader

        for epoch in range(num_epochs):
            start_epoch = time.time()
            
            train_loss = self._train_epoch()
            test_loss = self._test_epoch()

            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)

            # Generate reviews
            start_words = get_cluster_start_tokens(word2id)
            epoch_reviews = generate_cluster_reviews(
                start_words=start_words, 
                n_reviews=n_reviews,
                model=self.model,
                h_learnable=self.h_learnable,
                embedder=self.embedder,
                word2id=word2id,
                id2word=id2word,
                max_len=30,
                allow_unknowns=True,
                top_k=5
            )
            self.reviews.append(epoch_reviews)

            # Predict clusters of generated reviews
            clusters, reviews = clusterEval.process_gen_reviews(epoch_reviews)
            acc, conf = clusterEval.eval_clustering(reviews, clusters, fn_clustering='KMeansModel.joblib')
            self.cluster_accs.append(acc)
            self.cluster_confs.append(conf)

            # Print progress
            minutes_elapsed = round((time.time() - start_epoch) / 60)
            print(f"Epoch {len(self.train_losses)} \t Train-Loss = {round(float(train_loss), 2)}\t Test-Loss = {round(float(test_loss), 2)} \t Cluster-Acc.: {acc} \t Minutes: {minutes_elapsed}")

    def plot_loss(self):
        train_losses = np.array(self.train_losses)
        test_losses = np.array(self.test_losses)
        epochs = np.arange(start=1, stop=len(train_losses)+1, step=1)

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle("Train and Testloss")
        ax1.plot(epochs, train_losses, 'b-', label="Train")
        ax2.plot(epochs, test_losses, 'b-', label="Test")
        fig.show()

def word_to_model_input(word, embedder, word2id):
    """
    Args:
        word (str): e.g. "<SOR>"
        embedder (Embedder): 
    Return:
        A packed torch.Tensor containing the embedded word
    """
    x = torch.LongTensor([word2id[word]])
    x_embed = embedder.embed(x)
    x_embed = x_embed.view(1, 1, -1) 
    X_packed = pack_padded_sequence(x_embed, [1], batch_first=True, enforce_sorted=False)
    return X_packed

def predict(x_packed, model, h_learnable, h, batch_size, word2id, allow_unknowns, top_k=5):
    """
    Predicts the next word in a sequence.

    Args:
        X_packed (packed torch.Tensor): A single packed (1, M, E) input vector. 
        model (LSTM):                   trained model
        h_learnable (bool):             If True, the model has a learnable hidden/cell state
        h (tuple):                      tuple of (hidden_state, cell_state)
        allow_unkowns (bool)            (Dis-)allows predictions of <UNK> tokens
        top_k (int):                    sample a prediction from top_k predictions

    Return:
        pred ((int, hidden)): tuple containing (word_index, hidden_state)
    """
    x_packed = x_packed.to(device=device)
    model = model.to(device=device)
    
    model.eval()
    with torch.no_grad():
        if h_learnable:
            out, h = model.predict(x_packed, h, batch_size)
        else:
            out, h = model(x_packed, h)

        # Convert scores into probabilities
        probs = F.softmax(out, dim=1).data.cpu()
        probs = probs.squeeze()

        if not allow_unknowns:
            # Set prob of <UNK> to near zero. This prevents the model from
            # choosing <UNK> as its prediction.
            probs[word2id["<UNK>"]] = 1e-6
        
        # Get word indices and respective probabilities of the top_k predictions
        top_preds = np.asarray(np.argsort(-probs)[:top_k]) 
        top_preds_probs = np.asarray(probs[top_preds])

        # Choose a prediction by sampling from the top_k predictions
        if top_k == 1:
            return top_preds, h
        else:
            pred = np.random.choice(top_preds, p=top_preds_probs/top_preds_probs.sum())
            return pred, h

def generate_text(start_word, model, h_learnable, embedder, word2id, id2word, max_len, allow_unknowns, top_k=5):
    """
    Generate a single review.

    Args:
        start_word (string): E.g. "<SOR 1>"
        model (LSTM): Pytorch LSTM model
        h_learnable (bool): If True, the hidden/cell state is learnable
        embedder (Embedder): The embedder used to encode words
        word2id (dict)
        id2word (dict)
        max_len (int): The maximum number of words that the generated review can have
        allow_unknowns (bool): If True, allows <UNK> tokens in the generated review
        top_k (int): Samples from top_k predictions when predicting a word

    Returns:
        String of the review that was generated
    """
    model.eval()
    with torch.no_grad():
        # Try at most 10 times to generate a review that ends with "<EOR>".
        # If unsuccesful, output whatever review was generated last.
        trys = 0
        while trys <= 10:
            trys += 1
            word_ids = [word2id[start_word]]

            if h_learnable:
                h = None
            else:
                h = model.init_hidden(batch_size=1)
            
            while word_ids[-1] != word2id['<EOR>'] and len(word_ids) <= max_len:
                x = word_to_model_input(id2word[word_ids[-1]], embedder, word2id)
                y, h = predict(
                    x_packed=x,
                    model=model,
                    h_learnable=h_learnable,
                    h=h,
                    batch_size=1,
                    word2id=word2id,
                    allow_unknowns=allow_unknowns,
                    top_k=top_k
                )
                word_ids.append(y)
            
            if word_ids[-1] == word2id['<EOR>']:
                break
    words = [id2word[idx] for idx in word_ids]
    return " ".join(words)

def get_cluster_start_tokens(word2id):
    """
    Returns all special Start-of-Review tokens that have a cluster ID appended
    to them.
    """
    return [s for s in word2id.keys() if "<SOR " in s]

def generate_cluster_reviews(start_words, n_reviews, model, h_learnable, 
                            embedder, word2id, id2word, 
                            max_len=30, allow_unknowns=True, top_k=5):
    """
    A helper function that generates n_reviews for each word provided in
    start_tokens.

    Args:
        start_tokens (list): List containing strings of start_words
        n_reviews (int): Number of reviews to be generated per start_word
        model (LSTM): Pytorch LSTM model
        h_learnable (bool): If True, the hidden/cell state is learnable
        embedder (Embedder): The embedder used to encode words
        word2id (dict)
        id2word (dict)
        max_len (int): The maximum number of words that the generated review can have
        allow_unknowns (bool): If True, allows <UNK> tokens in the generated review
        top_k (int): Samples from top_k predictions when predicting a word

    Returns:
        List of lists, where the i'th inner list contains n_reviews generated
        reviews with the i'th start_word as the initial word.
    """
    reviews = []
    for start_word in start_words:
        n = 0
        while n < n_reviews:
            n += 1
            r = generate_text(
                start_word=start_word,
                model=model,
                h_learnable=h_learnable,
                embedder=embedder,
                word2id=word2id,
                id2word=id2word,
                max_len=max_len,
                allow_unknowns=allow_unknowns,
                top_k=top_k
            )
            reviews.append(r)
    return reviews

def save_reviews_in_gdrive(resource_folder, filename, reviews):
    """
    Writes multiple reviews to a CSV file.

    Args:
        reviews (list[str]): list of strings containing customer reviews
        filename (str):             e.g. "GeneratedReviews_15_04_2020.csv"
        resource_folder (str):      "/content/gdrive/My Drive/Customer Review Generation/Resources" 
    """
    import csv
    path = os.path.join(resource_folder, filename)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(reviews)

def load_model_from_checkpoint(checkpoint, h_learnable, lstmClass, modelClass, embedderClass):
    """
    Load a pytorch-model from gdrive and setup the wrapper class Model such 
    that one can conveniently continue training.

    Args:
        checkpoint (dict): loaded checkpoint after calling torch.load()
        lstmClass (LSTM): Class of the pytorch-lstm
        modelClass (Model)
        embedderClass (Embedder)
    
    Returns:
        Instance of Model that can be used for further training
    """
    # Get the previous states / checkpoints
    model_state_dict = checkpoint['model_state_dict']
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    params = checkpoint['other_params']

    # Rebuild the model
    lstm = lstmClass(input_size=params['input_size'], 
                output_size=params['output_size'],
                hidden_size=params['hidden_size'],
                num_layers=1).to(device=device)
    lstm.load_state_dict(model_state_dict)

    # Rebuild the optimizer
    optimizer = torch.optim.Adam(lstm.parameters())
    optimizer.load_state_dict(optimizer_state_dict)

    # Rebuild the embedder
    embedder = embedderClass(method=params['embed_method'], 
                        dict_size=params['output_size'],
                        embedding_dim=params['input_size'])

    # Rebuild the loss criterion
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Set up Model Wrapper Class
    model = modelClass(lstm, h_learnable, optimizer, criterion, embedder)
    model.train_losses = params['train_losses']
    model.test_losses = params['test_losses']
    model.reviews = params['reviews']
    model.cluster_accs = params['cluster_accs']
    model.cluster_confs = params['cluster_confs']

    return model