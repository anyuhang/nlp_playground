
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
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate
from keras.models import Sequential, Model

np.random.seed(1337)
SEQUENCE_LENGTH = 8
VOCAB_SIZE = 20000
EMBEDDING_DIM = 128
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

fn = '/Users/yan/qa-classifier/qa_recom_refQuestions.csv'
fn_test = '/Users/yan/qa-classifier/qa_recom_orig_min.csv'
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
X_train_data = []
X_train_doc_index = []
y_train_label = []
for doc in X_train:
    for i in xrange(0, len(doc[4])-1-SEQUENCE_LENGTH, STEP):
        X_train_data.extend([doc[4][i:i+SEQUENCE_LENGTH] ])
        X_train_doc_index.extend([doc_index[doc[0]]])
        y_train_label.extend([doc[3]])

X_train_data_arr = np.stack(X_train_data)
X_train_doc_idx_arr = np.stack(X_train_doc_index)
y_train_arr = np.stack(y_train_label)

# testing
X_test_data = []
X_test_doc_index = []
y_test_label = []
for doc in X_test:
    for i in xrange(0, len(doc[4])-1-SEQUENCE_LENGTH, STEP):
        X_test_data.extend([doc[4][i:i+SEQUENCE_LENGTH] ])
        X_test_doc_index.extend([doc_index[doc[0]]])
        y_test_label.extend([doc[3]])

X_test_data_arr = np.stack(X_test_data)
X_test_doc_idx_arr = np.stack(X_test_doc_index)
y_test_arr = np.stack(y_test_label)

# change label to categorical variable
label_train = to_categorical(np.asarray(y_train_label))
label_test = to_categorical(np.asarray(y_test_label))
print(label_train.shape, label_test.shape)

print(len(X_train_data), len(y_train_label), len(X_test_data), len(y_test_label))

# working model
model = Sequential()
embedding_layer_word = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=SEQUENCE_LENGTH,W_constraint=unitnorm(axis=1))
model.add(embedding_layer_word)
model.add(Flatten())
model.add(Dense(PA_CAT, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
model.fit(X_train_data, label_train, validation_data=(X_test_data, label_test),nb_epoch=1, batch_size=128)

# model
DOC_SIZE = len(doc_index)
word_emb = Sequential()
embedding_layer_word = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=SEQUENCE_LENGTH)
word_emb.add(embedding_layer_word)

word_input = Input(shape=(8,), dtype='int32', name='word_input')
word_emb = word_emb(word_input)

doc_emb = Sequential()
embedding_layer_doc = Embedding(input_dim=DOC_SIZE, output_dim=EMBEDDING_DIM, input_length=1)
doc_emb.add(embedding_layer_doc)

doc_input = Input(shape=(1,), dtype='int32', name='doc_input')
doc_emb = doc_emb(doc_input)

concat_input = concatenate([word_emb, doc_emb], axis=1)
model_input =Flatten()(concat_input)
dense = Dense(PA_CAT, activation='softmax')(model_input)
model = Model([word_input, doc_input], dense)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
model.fit([X_train_data_arr, X_train_doc_idx_arr], label_train, validation_data=([X_test_data_arr, X_test_doc_idx_arr], [label_test]),epochs=1, batch_size=128)
model.evaluate( [X_train_data_arr, X_train_doc_idx_arr], label_train, batch_size=128, verbose=1, sample_weight=None)
em = embedding_layer_word.get_weights()[0]

em_norm = LA.norm(em, axis=1)
#norm = np.sqrt(np.reduce_sum(np.square(em), 1, keep_dims=True))
em_n = em / em_norm.reshape((20000,1))
similarity = np.matmul(em_n, np.transpose(em_n))
i = 559
i= 358
top_k = 10
nearest = (-similarity[i, :]).argsort()[1:top_k+1]
log = 'Nearest to %s:' % index_word[i]
for k in range(top_k):
  close_word = index_word[nearest[k]]
  log = '%s %s,' % (log, close_word)

print(log)
