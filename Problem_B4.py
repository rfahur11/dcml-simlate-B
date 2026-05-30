# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import numpy as np

# Custom Callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        # Set target at 91.5% (0.915) to be safely above the 91% minimum threshold
        if(logs.get('accuracy') > 0.915 and logs.get('val_accuracy') > 0.915):
            print("\nTarget accuracy > 91.5% reached! Stopping training...")
            self.model.stop_training = True

def solution_B4():
    bbc = pd.read_csv('https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    # DO NOT CHANGE THIS CODE
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    # Extract texts and labels from the dataframe
    sentences = bbc['text'].values
    labels = bbc['category'].values

    # YOUR CODE HERE
    # Using "shuffle=False"
    training_sentences, validation_sentences, training_labels, validation_labels = train_test_split(
        sentences, labels, train_size=training_portion, shuffle=False
    )

    # Fit your tokenizer with training data
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)

    # Tokenize and pad the text data
    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(training_sequences, padding=padding_type, maxlen=max_length, truncating=trunc_type)

    validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
    validation_padded = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length, truncating=trunc_type)
    
    # You can also use a Tokenizer to encode your labels
    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels) # Fit on all labels so it recognizes every class
    
    # Subtract 1 because Tokenizer starts at 1, while sparse_categorical_crossentropy expects labels starting at 0
    training_label_seq = np.array(label_tokenizer.texts_to_sequences(training_labels)).flatten() - 1
    validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels)).flatten() - 1

    model = tf.keras.Sequential([
        # YOUR CODE HERE.
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    # Make sure you are using "sparse_categorical_crossentropy" as the loss function
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Initialize callback
    callbacks = myCallback()

    model.fit(
        training_padded,
        training_label_seq,
        epochs=50, # Set epochs long enough; let the callback stop training early
        validation_data=(validation_padded, validation_label_seq),
        callbacks=[callbacks],
        verbose=1
    )

    return model

# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B4()
    model.save("model_B4.h5")