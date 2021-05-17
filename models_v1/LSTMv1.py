import pandas as pd
import numpy as np
import fasttext
from tqdm import tqdm
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional, Dropout
import gc
import pickle


fasttext_model = fasttext.load_model('cc.en.300.bin')


# tokenize, create embedding and save
def prepare_data(scenario, MAX_LEN=256, NUM_WORDS=171000):
    train = pd.read_csv(f'Aggression{scenario}/train.tsv', sep='\t')
    dev = pd.read_csv(f'Aggression{scenario}/dev.tsv', sep='\t')
    test = pd.read_csv(f'Aggression{scenario}/test.tsv', sep='\t')
    tokenizer = Tokenizer(num_words = NUM_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(train.sentence.tolist() + dev.sentence.tolist() + test.sentence.tolist())
    train_sequences = tokenizer.texts_to_sequences(train.sentence.tolist())
    train_padded = pad_sequences(train_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    dev_sequences = tokenizer.texts_to_sequences(dev.sentence.tolist())
    dev_padded = pad_sequences(dev_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    test_sequences = tokenizer.texts_to_sequences(test.sentence.tolist())
    test_padded = pad_sequences(test_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 300))
    for word, i in tqdm(tokenizer.word_index.items()):
        embedding_matrix[i] = fasttext_model.get_word_vector(word)
    scenario_data = {'train_X': train_padded, 'dev_X': dev_padded, 'test_X': test_padded,
                     'train_y': train.target.to_numpy(), 'dev_y': dev.target.to_numpy(),
                     'test_y': test.target.to_numpy(),
                     'embedding_matrix': embedding_matrix}
    with open(f'Aggression{scenario}/data.pkl', 'wb') as file:
        pickle.dump(scenario_data, file)


def check_lstm(scenario='S1'):
    with open(f'Aggression{scenario}/data.pkl', 'rb') as file:
        data = pickle.load(file)
    lstm = Sequential()
    lstm.add(Embedding(data['embedding_matrix'].shape[0], 300,
                       weights=[data['embedding_matrix']], trainable=False, mask_zero=True))
    lstm.add(LSTM(32, dropout=0.5, recurrent_dropout=0))
    lstm.add(Dense(1, activation="sigmoid"))
    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
    lstm.compile(loss='binary_crossentropy', optimizer=opt,
                 metrics=[tf.keras.metrics.Recall(name="recall"),
                          tf.keras.metrics.Precision(name="prec")])
    es = tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
    history = lstm.fit(data['train_X'], data['train_y'], epochs=50, batch_size=1000,
                       validation_data=(data['dev_X'], data['dev_y']), callbacks=[es])
    preds = lstm.predict(data['test_X'])
    result = classification_report(data['test_y'], np.round(preds), output_dict=True)
