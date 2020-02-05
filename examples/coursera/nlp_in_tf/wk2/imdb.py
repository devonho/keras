import numpy as np

import tensorflow_datasets as tfds
import tensorflow

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, GlobalAveragePooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ProgbarLogger, Callback

MAX_LENGTH = 120
TRUNC_TYPE = 'post'
OOV_TOK = '<OOV>'
VOCAB_SIZE = 10000
EMBEDDING_DIM = 16
EPOCHS = 150
BATCH_SIZE = 2048
VERBOSE = 2

def load_imdb_data():
    imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

    train_data, test_data = imdb['train'], imdb['test']

    training_sentences = []
    training_labels = []

    testing_sentences = []
    testing_labels = []

    # str(s.tonumpy()) is needed in Python3 instead of just s.numpy()
    for s, l in train_data:
        training_sentences.append(str(s.numpy()))
        training_labels.append(l.numpy())

    for s, l in test_data:
        testing_sentences.append(str(s.numpy()))
        testing_labels.append(l.numpy())

    training_labels_final = np.array(training_labels)
    testing_labels_final = np.array(testing_labels)

    return training_sentences, training_labels_final, testing_sentences, testing_labels_final

def preprocessing(training_sentences, testing_sentences):
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOK)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index

    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(training_sequences, maxlen=MAX_LENGTH, truncating=TRUNC_TYPE)

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=MAX_LENGTH, truncating=TRUNC_TYPE)

    #training_sequences = np.array(training_sequences)
    #testing_sequences = np.array(testing_sequences)


    return training_padded, testing_padded

def make_model():
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
        Flatten(),
        #GlobalAveragePooling1D(),
        Dense(6, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    model.summary()

    return model

def train_model(model, X_train, Y_train, X_valid, Y_valid):

    class CallBackLogger(Callback):
        def on_epoch_begin(self, epoch, logs=None):
            print('Epoch began ' + str(epoch))

    callbacks = [EarlyStopping(monitor='val_loss', patience=5), ProgbarLogger(), CallBackLogger()]


    history = model.fit(
        X_train,
        Y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=VERBOSE,
        callbacks=callbacks,
        validation_data=(X_valid, Y_valid))    
    return model, history



if __name__ == '__main__':
    print(tensorflow.__version__)
    training_sentences, training_labels_final, testing_sentences, testing_labels_final = load_imdb_data()
    training_sequences, testing_sequences = preprocessing(training_sentences, testing_sentences)
    model = make_model()
    model, history = train_model(model, training_sequences, training_labels_final, testing_sequences, testing_labels_final)
    model.save('imdb_sentiment.h5')

    