import json

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_sarcasm_data():
    with open('data/Sarcasm_Headlines_Dataset.json') as f:
        data = json.load(f)
    return data


def do_tokenize(texts : list):
    tokenizer = Tokenizer(oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)

    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, padding = 'post')
    return word_index, padded
    



if __name__ == '__main__':
    data = load_sarcasm_data()
    texts = [e['headline'] for e in data]
    labels = [e['is_sarcastic'] for e in data]
    word_index, padded = do_tokenize(texts)

    print(len(word_index))