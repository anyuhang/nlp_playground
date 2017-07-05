
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
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate, LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Sequential, Model
from keras.constraints import unitnorm
import keras.backend as K
from keras.layers.normalization import BatchNormalization
from keras import regularizers

os.chdir('/home/ubuntu/yan/d2v_supervised')

np.random.seed(1337)
SEQUENCE_LENGTH = 8
MAX_SEQUENCE_LENGTH = 1000
VOCAB_SIZE = 40000
EMBEDDING_DIM = 100
PA_CAT = 51
STEP = 4

def read_sentences(handle):
 for l in handle:
   if l.strip():
     yield l.strip()

def process(sentence):
  s = sentence.split('|')
  qid = int(s[0].strip(","))
  sid = int(s[1].strip(","))
  question = splitWords(removePunctations(s[2].strip(",")))
  #question_parsed = " ".join(str(word) for word in question if word not in stoplist)
  question_parsed = " ".join(str(word) for word in question)
  return qid,sid,question_parsed

def readfile(fn):
  sen = []
  for sentence in read_sentences(open(fn)):
      a,b,c = process(sentence)
      sen.append([a,b,c])
  return sen

def splitWords(sentence):
  return sentence.lower().split(' ')

def removePunctations(text):
  return ''.join(t for t in text if t.isalnum() or t == ' ')

def writefile(inputFile, outputFile):
  output = open(outputFile, 'w')
  for sentence in read_sentences(open(inputFile)):
      a,b,c = process(sentence)
      output.write("%d,%d,%s\n" % (a,b,c))
  output.close()

def readCleanFile(fn):
  sen = []
  for sentence in read_sentences(open(fn)):
      a,b,c = sentence.split(',')
      sen.append([a,b,c])
  return sen

#fn = '/Users/yan/qa-classifier/qa_recom_refQuestions.csv'
#fn_test = '/Users/yan/qa-classifier/qa_recom_orig_min.csv'
fn = 'qa_recom_refQuestions.csv'
fn_test = 'qa_recom_orig_min.csv'
sen = readfile(fn)
sen_test = readfile(fn_test)

texts = [i[2] for i in sen]
tokenizer = Tokenizer(VOCAB_SIZE, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True, split=" ", char_level=False)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
index_word = dict(zip(word_index.values(), word_index.keys()))
print('Found %s unique tokens.' % len(word_index))

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

# add word_idx
for i in range(len(sequences)):
  sen[i].append(sequences[i])

# sampling
z = [i[3] for i in sen]
X_train, X_test, y_train, y_test = train_test_split(sen, z, test_size=0.1, stratify=z, random_state=42)

# training

#lens = np.array(map(len, X_train_data_ar))
#(lens.max(), lens.min(), lens.mean())

X_train_data = []
X_train_doc_index = []
y_train_label = []
for doc in X_train:
    X_train_data.extend([doc[4]])
    X_train_doc_index.extend([doc_index[doc[0]]])
    y_train_label.extend([doc[3]])

X_train_data_arr = pad_sequences(X_train_data, maxlen=MAX_SEQUENCE_LENGTH, value=0)
X_train_doc_idx_arr = np.stack(X_train_doc_index)
#y_train_arr = np.stack(y_train_label)

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
#y_test_arr = np.stack(y_test_label)

# change label to categorical variable
label_train = to_categorical(np.asarray(y_train_label))
label_test = to_categorical(np.asarray(y_test_label))
print(label_train.shape, label_test.shape)

print(len(X_train_data), len(y_train_label), len(X_test_data), len(y_test_label))

# Classification Model with 1 Conv1D and MaxPooling1D

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_layer_word = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH,embeddings_constraint=unitnorm(axis=1))
embedded_sequences = embedding_layer_word(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(label_train.shape[1], activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
model.fit(X_train_data_arr, label_train, validation_data=(X_test_data_arr, label_test),epochs=1, batch_size=128)
