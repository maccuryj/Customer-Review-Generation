# Customer Reviews: Conditional Generative Models On Clustered Text Data
---

Pattern Analysis and Machine Intelligence Project by Nils Gandlau, Jean Maccury, Fabian Ulmer

The goal of generative models is to find patterns in data, which can ultimately be recreated in a generated output. Recurrent Neural Networks (RNN) implementing a Long Short-Term Memory (LSTM) architecture have gained increasing popularity in this area, specifically in learning from and generating new text in accordance with the input data. In this project, our aim is to increase the influence one can take on the output of such a model. To this end, working with the Amazon Customer Review Dataset, we use a novel Text Embedding approach to represent our input data. The resulting text embeddings are then clustered by means of a minibatch KMeans model. Finally, we train an LSTM on the generation of such reviews. The trained LSTM ultimately serves as a generative model, which produces reviews from a specific cluster when setting an appropriate token. In sum, we provide an end to end approach for the conditional generation of text based on subsets of a complete corpus.

This project serves as a final group submission for the course "Pattern Analysis and Machine Learning" at Goethe-University Frankfurt.

