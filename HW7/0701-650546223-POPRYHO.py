import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
import matplotlib.pyplot as plt

# Define the NameGeneratorLSTM class
class NameGeneratorLSTM:
    def __init__(self, input_size, hidden_size, num_layers):
        self.model = Sequential()
        for i in range(num_layers):
            return_sequences = True
            if i == 0:
                self.model.add(LSTM(hidden_size, return_sequences=return_sequences,
                                    input_shape=(None, input_size)))
            else:
                self.model.add(LSTM(hidden_size, return_sequences=return_sequences))
        self.model.add(Dense(input_size, activation='softmax'))

    def compile(self, optimizer, loss):
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, X, Y, epochs, batch_size, verbose):
        return self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def save(self, filename):
        self.model.save(filename)

# Load and preprocess data
with open('names.txt', 'r') as file:
    names = file.read().lower().splitlines()

EON_CHAR = '`'
chars = set(''.join(names)) | {EON_CHAR}
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

seq_length = 11
dataX = []
dataY = []

for name in names:
    name_padded = name + EON_CHAR * (seq_length - len(name))
    seq_in = [char_to_int[char] for char in name_padded]
    seq_out = [char_to_int[char] for char in name[1:] + EON_CHAR * (seq_length - len(name) + 1)]
    dataX.append(seq_in)
    dataY.append(seq_out)

X = np.zeros((len(dataX), seq_length, len(chars)), dtype=bool)
Y = np.zeros((len(dataY), seq_length, len(chars)), dtype=bool)

for i, sequence in enumerate(dataX):
    for t, char_index in enumerate(sequence):
        X[i, t, char_index] = 1

for i, sequence in enumerate(dataY):
    for t, char_index in enumerate(sequence):
        Y[i, t, char_index] = 1

# Model parameters
input_size = len(chars)
hidden_size = 256
num_layers = 3
name_generator = NameGeneratorLSTM(input_size, hidden_size, num_layers)
name_generator.compile(optimizer='adam', loss='categorical_crossentropy')

model_filename = '0702-IDNumber-LastName.ZZZ'

if not os.path.exists(model_filename):
    history = name_generator.fit(X, Y, epochs=100, batch_size=64, verbose=1)
    name_generator.save(model_filename)
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()
