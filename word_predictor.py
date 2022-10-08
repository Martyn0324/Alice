from PyPDF2 import PdfFileReader as PFR
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import torch
import numpy as np
import pandas as pd
import re

FILE_PATH = 'C:/Users/giova/OneDrive/Área de Trabalho/Faster than the Flame.txt'

file = open(FILE_PATH, 'r')

text = []

for i in file:
    text.append(str(i)) # Extracting all words into a single list

file.close()

text = [x.lower() for x in text]
text = [re.sub('[^a-z0-9\s\()]', '', x) for x in text]

text_total = ' '.join(text)

text_total = ' '.join(text_total.split()) # Removing double-spaces.

words_text = list(text_total.split(' ')) # NLTK, Spacy...in the end, they just make things more complicated...perhaps except for tokenizing.


class WordDataset(object):
    def __init__(self, words):
        self.words = words

        self.word2idx = {}
        self.idx2word = []
        self.sequence_length = None
        self.word_sequence = []
        self.token_sequence = []

        self.idx_values = []

    def create_dictionary(self):
        for word in self.words:
            if word not in self.word2idx:
                self.idx2word.append(word)
                self.word2idx[word] = len(self.idx2word) - 1

    def normalize(self): # Data within [-1,1] is better than integers from [0, infinite[.
        all_values = self._get_min_max()

        all_values = all_values

        for word, value in self.word2idx.items():

            scaled_value = (value - np.min(all_values))*2.0 / (np.max(all_values) - np.min(all_values))-1.0

            self.word2idx[word] = scaled_value

    def create_sequence(self, sequence_length):
        self.sequence_length = sequence_length # Storing sequence length for posterior use
        for i in range(sequence_length, len(self.words)):
            self.word_sequence.append(self.words[i-sequence_length:i])


    def tokenize_sequence(self):
        tokens = []
        for sequence in self.word_sequence:
            for word in sequence:
                token = self.word2idx.get(word)
                tokens.append(token)
        
        for i in range(self.sequence_length, len(self.words)):
            self.token_sequence.append(tokens[i-self.sequence_length:i])

        self.token_sequence = np.array(self.token_sequence) # Ready to be used

    def detokenize(self, tokens):
        # Unfortunately, it seems detokenizing isn't really something that people tend to worry about...at least when dealing with word prediction.
        # Since we're not dealing with integers, we'll have to use another algorithm, K-Nearest Neighbor

        tokens = tokens.reshape(-1, 1) # Reshaping so KNN can accept our data

        values = list(self.word2idx.values()) # Extracting tokens to fit our KNN
        values = np.array(values).reshape(-1,1)
        knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(values)

        _, index = knn.kneighbors(tokens) # The distance is irrelevant, so extracting the indexes only

        keys = list(self.word2idx.keys())

        words = []

        for subarray in index:
            for i in subarray:
                words.append(keys[i])

        word_sequence = []

        for i in range(self.sequence_length, len(words)):
            word_sequence.append(words[i-self.sequence_length:i])

        return words, word_sequence


    def __len__(self):
        return len(self.idx2word)
      
    def __getitem__(self, idx):

        features = self.token_sequence[idx]

        return features

    def _get_min_max(self):

        for _, values in self.word2idx.items():

            self.idx_values.append(values)

        self.idx_values = np.array(self.idx_values)

        return self.idx_values
        

        
dataset = WordDataset(words_text)

dataset.create_dictionary()

#print(dataset.word2idx)

dataset.normalize()

#print(dataset.word2idx)

dataset.create_sequence(sequence_length=5)

#print(dataset.word_sequence)

dataset.tokenize_sequence()

#print(dataset.token_sequence)

dataframe = pd.DataFrame(dataset.token_sequence) # Creating a DataFrame is a good way to organize and check our data.

X = dataframe.drop(dataframe.columns[4], axis=1)
y = dataframe[dataframe.columns[4]] # It also makes selecting the labels easier

X = X.to_numpy()
y = y.to_numpy()

X = torch.from_numpy(X).cuda()
y = torch.from_numpy(y).cuda()

#ntokens = len(dataset.word2idx) # We won't be using this

#print(ntokens)

class LSTMModel(torch.nn.Module):
    def __init__(self, ninp):
        super(LSTMModel, self).__init__()

        #self.embed = torch.nn.Embedding(ntoken, ninp) # No Embedding, since we won't be using one-hot encoding.
        self.lstm = torch.nn.LSTM(ninp, 10, 5, batch_first=True, bias=False)
        self.neuron1 = torch.nn.Linear(10, 50, bias=False)
        self.neuron2 = torch.nn.Linear(50, 1, bias=False)
        self.tanh = torch.nn.Tanh() # Using Tanh, since our data is within [-1, 1]

    def forward(self, input):

        #x = self.embed(input)

        x, _ = self.lstm(input)

        x = self.neuron1(x)
        x = self.neuron2(x)

        output = self.tanh(x)

        return output
      
      
model = LSTMModel(4).cuda().double() # X is composed by 4 tokens sequences. We also have to convert the model parameters to Double, as it'll be Float by default.

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss = torch.nn.MSELoss() # And this is how we can discard embedding.

EPOCHS = 100
BATCH_SIZE = 10

for epoch in range(EPOCHS):
    for batch in range(BATCH_SIZE, len(X)):
        model.zero_grad()

        input = X[batch-10:batch]

        output = model(input)

        labels = y[batch-10:batch]

        output = output.view(-1)

        cost = loss(output, labels)

        cost.backward()

        optimizer.step()

    if epoch % 10 == 0:
        print(f"{epoch}/{EPOCHS}\t Current Loss: {cost.item()}")


input = input.cpu().numpy() # Getting last input

for i in input:
    input_word, _ = dataset.detokenize(i) # We won't use the word_sequence output here...as our sequence_length is 5 and our input is made of 4 length sequences

labels = labels.cpu().numpy()

for i in labels:
    label, _ = dataset.detokenize(i)

output = output.detach().cpu().numpy()

for i in output:
    output, _ = dataset.detokenize(i)

final = pd.DataFrame(input_word)
final = final.T

final = final.assign(Labels=label)

final = final.assign(Predicted=output)

print(final)
