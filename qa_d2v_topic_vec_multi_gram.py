
from sklearn.model_selection import train_test_split
from __future__ import print_function
import os, sys, collections
import numpy as np
from numpy import linalg as LA
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers.core import Dense, Flatten
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate, LSTM, Merge
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Sequential, Model
from keras.constraints import unitnorm
import keras.backend as K
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from nlp_utility import *


'''
# laptop version
kw_file = '/Users/yan/Code/avvo_nlp/qa_toad_kw.csv'
fn = '/Users/yan/qa-classifier/qa_recom_refQuestions.csv'
fn_test = '/Users/yan/qa-classifier/qa_recom_orig_min.csv'
'''
## AWS version
kw_file = 'qa_toad_kw.csv'
fn = 'qa_recom_refQuestions.csv'
fn_test = 'qa_recom_orig_min.csv'
sen = readfile(fn)

kw_list = get_kw_list(kw_file)

def tokenizer(texts):
    sentences = []
    for sen_cur in texts:
        words = []
        for i in sen_cur.split(' '):
            if len(i) > 0:
                words.append(i)
        sentences.append(words)
    return sentences

texts = [i[2] for i in sen]
sentences = tokenizer(texts)

unigram = build_unigram(sentences)
bigram = [i[0] for i in kw_list if i[1]==2]
trigram = [i[0] for i in kw_list if i[1]==3]
fourgram = [i[0] for i in kw_list if i[1]==4]
count = [('UNK', -1)]
count = count_words(count, fourgram, VOCAB_SIZE = len(fourgram))
count = count_words(count, trigram, VOCAB_SIZE = len(trigram))
count = count_words(count, bigram, VOCAB_SIZE = len(bigram))
count = count_words(count, unigram, VOCAB_SIZE = 60000)
dictionary, reverse_dictionary = build_dictionary(count)
sequences = texts_to_sequences(sentences, dictionary, span=4)
print(len(sequences[2]),len(sequences))

# add doc_idx
X = [i[0] for i in sen]
from collections import defaultdict
doc_index = {}
for i in X:
    if i not in doc_index:
        doc_index[i] = len(doc_index)

index_doc = dict(zip(doc_index.values(), doc_index.keys()))

# add pa_idx
y = [i[1] for i in sen]
pa_keep = [ i[0] for i in collections.Counter(y).most_common(50)]
pa_dict = {}
for pa_idx, pa in enumerate(pa_keep):
    pa_dict[pa] = pa_idx

for i in sen:
    if i[1] in pa_dict:
        i.append(pa_dict[i[1]])
    else:
        i.append(50)

# configuration
np.random.seed(1337)
VOCAB_SIZE = len(dictionary)
EMBEDDING_DIM = 64
PA_CAT = 51
MAX_SEQUENCE_LENGTH = 1000

# add word_idx
for i in range(len(sequences)):
  sen[i].append(sequences[i])

# sampling
z = [i[3] for i in sen]
X_train, X_test, y_train, y_test = train_test_split(sen, z, test_size=0.1, stratify=z, random_state=42)

# training
X_train_data = []
X_train_doc_index = []
y_train_label = []
for doc in X_train:
    X_train_data.extend([doc[4]])
    X_train_doc_index.extend([doc_index[doc[0]]])
    y_train_label.extend([doc[3]])

X_train_data_arr = pad_sequences(X_train_data, maxlen=MAX_SEQUENCE_LENGTH, value=0)
X_train_doc_idx_arr = np.stack(X_train_doc_index)

# testing
X_test_data = []
X_test_doc_index = []
y_test_label = []
for doc in X_test:
    X_test_data.extend([doc[4]])
    X_test_doc_index.extend([doc_index[doc[0]]])
    y_test_label.extend([doc[3]])

X_test_data_arr = pad_sequences(X_test_data, maxlen=MAX_SEQUENCE_LENGTH, value=0)
X_test_doc_idx_arr = np.stack(X_test_doc_index)

# change label to categorical variable
label_train = to_categorical(np.asarray(y_train_label))
label_test = to_categorical(np.asarray(y_test_label))
print(label_train.shape, label_test.shape)
print(len(X_train_data), len(y_train_label), len(X_test_data), len(y_test_label))

# w2v model
# w2v with multiple layers
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_layer_word = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH,embeddings_constraint=unitnorm(axis=1))
embedded_sequences = embedding_layer_word(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
Dropout(0.2)
BatchNormalization()
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
Dropout(0.2)
BatchNormalization()
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)  # global max pooling
Dropout(0.2)
BatchNormalization()
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(label_train.shape[1], activation='softmax')(x)
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
model.fit(X_train_data_arr, label_train, validation_data=(X_test_data_arr, label_test),epochs=1, batch_size=128)
model.save_weights('ft1.h5')

# fewer layers
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_layer_word = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH,embeddings_constraint=unitnorm(axis=1))
embedded_sequences = embedding_layer_word(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Flatten()(x)
x = Dense(640, activation='relu')(x)
Dropout (0.25)
preds = Dense(label_train.shape[1], activation='softmax')(x)
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
model.fit(X_train_data_arr, label_train, validation_data=(X_test_data_arr, label_test),epochs=3, batch_size=128)

# combined layers
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_layer_word = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH,embeddings_constraint=unitnorm(axis=1))
Dropout (0.2)
embedded_sequences = embedding_layer_word(sequence_input)
convs = [ ]
for fsz in range (2, 5):
    c = Conv1D(64, fsz, border_mode='same', activation="relu")(embedded_sequences)
    c = MaxPooling1D()(c)
    c = Flatten()(c)
    convs.append(c)

x = concatenate(convs)
Dropout (0.2)
BatchNormalization()
x = Dense(128, activation='relu')(x)
preds = Dense(label_train.shape[1], activation='softmax')(x)
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['acc'])
model.fit(X_train_data_arr, label_train, validation_data=(X_test_data_arr, label_test),epochs=3, batch_size=128)


# tensorfow version
VOCAB_SIZE = len(dictionary)
EMBEDDING_DIM = 64
PA_CAT = 51
MAX_SEQUENCE_LENGTH = 1000
batch_size = 128
depth = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, 1, 51))
  tf_valid_dataset = tf.placeholder(tf.float32, shape=(batch_size, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM))
  tf_test_dataset = tf.placeholder(tf.float32, shape=(batch_size, 1, 51))

  # Variables.
  embeddings = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBEDDING_DIM], -1.0, 1.0))
  weights = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBEDDING_DIM],stddev=1.0 / math.sqrt(EMBEDDING_DIM)))
  biases = tf.Variable(tf.zeros([VOCAB_SIZE]))

  embed = tf.zeros([batch_size, EMBEDDING_DIM])
  for j in range(1000):
    embed += tf.nn.embedding_lookup(embeddings, train_dataset[:, j])
    embed = embed

  # Model.
  def model(data):
    conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
    conv = tf.nn.conv1d(data, filters=128, stride=5, padding='SAME')
    hidden = tf.nn.relu(conv + layer1_biases)
    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    return tf.matmul(hidden, layer4_weights) + layer4_biases

  # Training computation.
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model(tf_test_dataset))
