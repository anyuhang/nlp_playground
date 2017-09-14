
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

kw_file = '/home/ec2-user/src/corpus/questions.csv'
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

index_pa = dict(zip(pa_dict.values(), pa_dict.keys()))

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
x = MaxPooling1D(27)(x)  # global max pooling
Dropout(0.2)
BatchNormalization()
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(label_train.shape[1], activation='softmax')(x)
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
model.fit(X_train_data_arr, label_train, validation_data=(X_test_data_arr, label_test),epochs=3, batch_size=128)
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

pred_output = model.predict((X_test_data_arr, batch_size=128)


# combined layers
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_layer_word = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH,embeddings_constraint=unitnorm(axis=1))
Dropout (0.2)
embedded_sequences = embedding_layer_word(sequence_input)
convs = [ ]
for fsz in range (3, 6):
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

###
from sklearn.manifold import TSNE
from matplotlib import pylab
num_points = 400
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
two_d_embeddings = tsne.fit_transform(em_word[500:num_points+500, :])
def plot(embeddings, labels):
  assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
  pylab.figure(figsize=(15,15))  # in inches
  for i, label in enumerate(labels):
    x, y = embeddings[i,:]
    pylab.scatter(x, y)
    pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                   ha='right', va='bottom')
  pylab.show()

words = [reverse_dictionary[i] for i in range(500, num_points+500)]
plot(two_d_embeddings, words)
np.save('tsne_2d', two_d_embeddings)
two_d_embeddings=np.load('tsne_2d.npy')

import csv
with open("tsne_words.csv", "w") as f:
    wr = csv.writer(f,delimiter="\n")
    wr.writerow(words)

import csv
with open('tsne_words.csv', 'rb') as f:
    reader = csv.reader(f)
    words = list(reader)

for i in words:
    word.extend(i)

plot(two_d_embeddings, word)
### check results by PA categories
### y_train_label is true label; y_hat is [rows, 51] matrix
https://stackoverflow.com/questions/6910641/how-to-get-indices-of-n-maximum-values-in-a-numpy-array
https://stackoverflow.com/questions/30332908/n-largest-values-in-each-row-of-ndarray -- good
https://stackoverflow.com/questions/6252280/find-the-most-frequent-number-in-a-numpy-vector -- bin count
y_hat = model.predict(X_train_data_arr, batch_size=128)
>>> y_hat_idx = y_hat.argmax(axis=1)
>>> y_hat_idx.shape
(115616,)
>>> y_hat_idx[:4]
array([50,  0,  4,  9])
>>> y = np.asarray(y_train_label)
>>> y.shape
(115616,)
>>> y[:4]
array([50,  2,  2, 34])
>>> y_cat=np.where(y==50)
>>> y_hat_idx[y_cat[0]][:10]
array([50,  2, 50,  5, 50, 50, 50,  9, 50, 50])
>>> len(y_hat_idx[y_cat[0]])
6105
>>> sum(y_hat_idx[y_cat[0]]==50)
3366
>>> a = np.array([9, 4, 4, 3, 3, 10, 0, 4, 6, 0])
>>> ind = np.argpartition(a, -4)[-4:]
>>> ind
array([1, 5, 8, 0])
>>> a[ind]
array([4, 9, 6, 9])
>>> ind[np.argsort(a[ind])]
array([1, 8, 5, 0])
>>> ind[np.argsort(a[ind])][::-1]
array([0, 5, 8, 1])

from numpy import linalg as LA
a = np.array([[0, 3, 4, 2, 5],
              [4, 2, 6, 3, 1],
              [2, 1, 1, 8, 8],
              [6, 6, 3, 2, 6]])
a_norm = LA.norm(a, axis=1)
a_n = a / a_norm.reshape((a.shape[0],1))
>>> sorted_row_idx = np.argsort(a, axis=1)[:,2::]
>>> sorted_row_idx
array([[1, 2, 4],
       [3, 0, 2],
       [0, 3, 4],
       [0, 1, 4]])
sorted_row_idx = np.argsort(a_n, axis=1)[:,-2::][:,::-1]
sorted_row_idx
array([[4, 2, 1],
       [2, 0, 3],
       [4, 3, 0],
       [4, 1, 0]])

a=np.array([[ 5,  4,  3,  2,  1],
            [10,  9,  8,  7,  6]])
b=np.argpartition(a,3)
b[:,-3:][:,::-1]
array([[0, 1, 2],
       [0, 1, 2]])


## top 1 prediction
acc_list = []
for i in xrange(51):
    y_cat=np.where(y==i)
    acc_list.append([i, len(y_cat[0]), sum(y_hat_idx[y_cat[0]]==i),
    float(sum(y_hat_idx[y_cat[0]]==i))/max(len(y_cat[0]),1)])

for i in acc_list:
    print(index_pa[i[0]], i[1], i[2], i[3])

## top 3 predictions
y_hat_2 = np.sort(y_hat, axis=1)[:,-2::][:,::-1]
y_hat_idx_2 = np.argsort(y_hat, axis=1)[:,-2::][:,::-1]
acc_list = []
for i in xrange(51):
i = 6
y_cat=np.where(y==i)
row_idx = y_cat[0]
num_rows = row_idx.shape[0]
count = np.bincount(y_hat_idx_2[row_idx,0])

y_hat[row_idx,0]
acc_list.append([i, index_pa[i],
# find top 3 largest customer_category
# find top 3 largest customer_category count
np.argsort(-count)[:3],
-np.sort(-count)[:3],
# first pred is correct
sum(y_hat_idx_2[row_idx,0]==i),
# second pred is correct
sum(y_hat_idx_2[row_idx,1]==i),
# first pred correct pct
float(sum(y_hat_idx_2[row_idx,0]==i))/max(num_rows,1),
# second pred correct pct
float(sum(y_hat_idx_2[row_idx,1]==i))/max(num_rows,1)])

for i in acc_list:
    print(i)
