import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd

# Load the dataset in chunks
def load_data_in_chunks(file_path, chunk_size=5000):  # Reduce chunk size
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        yield chunk['content']

# Tokenize and preprocess the data dynamically
def preprocess_data_generator(file_path, tokenizer, max_sequence_len, batch_size):
    for chunk in load_data_in_chunks(file_path):
        texts = " ".join(chunk).split('\n')
        sequences = []
        
        for line in texts:
            token_list = tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                sequences.append(n_gram_sequence)

                if len(sequences) >= batch_size:
                    sequences = pad_sequences(sequences, maxlen=max_sequence_len, padding='pre')
                    X = sequences[:, :-1]
                    y = sequences[:, -1]
                    y = tf.keras.utils.to_categorical(y, num_classes=len(tokenizer.word_index) + 1)
                    yield X, y
                    sequences = []  # Reset batch

# Initialize tokenizer and determine max sequence length
def initialize_tokenizer(file_path, max_sequence_limit=100):  # Limit sequence length
    tokenizer = Tokenizer()
    max_sequence_len = 0

    for chunk in load_data_in_chunks(file_path):
        texts = " ".join(chunk).split('\n')
        tokenizer.fit_on_texts(texts)

        for line in texts:
            token_list = tokenizer.texts_to_sequences([line])[0]
            max_sequence_len = min(max_sequence_limit, max(max_sequence_len, len(token_list)))  # Limit seq length

    return tokenizer, max_sequence_len

# File path to the dataset
file_path = 'archive/stories.csv'

# Initialize tokenizer and determine max sequence length
tokenizer, max_sequence_len = initialize_tokenizer(file_path)
total_words = len(tokenizer.word_index) + 1

# Build the model
model = Sequential()
model.add(Embedding(total_words, 50, input_length=max_sequence_len-1))
model.add(SimpleRNN(100, activation='relu'))
model.add(Dense(total_words, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save the model periodically
checkpoint_callback = ModelCheckpoint(
    filepath='word_rnn_model.h5',  # Save the model to this file
    save_best_only=True,           # Save only the best model
    monitor='loss',                # Monitor validation loss
    verbose=1
)

# Train the model using the data generator
batch_size = 512  # Reduce batch size to fit in memory
steps_per_epoch = 100  # Adjust number of batches per epoch

model.fit(
    preprocess_data_generator(file_path, tokenizer, max_sequence_len, batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=150,
    verbose=1,
    callbacks=[checkpoint_callback]
)
