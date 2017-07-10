# step 1: get unigram and bigram
# each doc is a list
count = [('UNK', -1)]
def build_bigram(sentences):
    bigram = []
    span = 2
    for sen_cur in sentences:
        for i in range(len(sen_cur) - span + 1):
            if (len(sen_cur[i]) > 0 and len(sen_cur[i+1])) > 0:
        	    bigram.append(sen_cur[i] + '_' + sen_cur[i+1])
    return bigram

def build_unigram(sentences):
    unigram = []
    for sen_cur in sentences:
        for i in sen_cur:
            if len(i) > 0:
                unigram.append(i)
    return unigram

# step 2: get dictionary
def count_words(words, VOCAB_SIZE=20000):
    count.extend(collections.Counter(words).most_common(VOCAB_SIZE - 1))
    return count

def build_dictionary(count):
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return  dictionary, reverse_dictionary

# step 3: get sequence of data_line using dictionary
def texts_to_sequences(sentences, span=4):
    data = []
    span = span
    for sen_cur in sentences:
        data_line = []
        i = 0
        while i < len(sen_cur):
            if i < (len(sen_cur) - 3) and (sen_cur[i] + '_' + sen_cur[i+1] + '_' + sen_cur[i+2] + '_' + sen_cur[i+3]) in dictionary:
                data_line.append(dictionary[(sen_cur[i] + '_' + sen_cur[i+1] + '_' + sen_cur[i+2] + '_' + sen_cur[i+3])])
                i = i + 4
            elif i < (len(sen_cur) - 2) and (sen_cur[i] + '_' + sen_cur[i+1] + '_' + sen_cur[i+2]) in dictionary:
                data_line.append(dictionary[(sen_cur[i] + '_' + sen_cur[i+1] + '_' + sen_cur[i+2])])
                i = i + 3
            elif i < (len(sen_cur) - 1) and (sen_cur[i] + '_' + sen_cur[i+1]) in dictionary:
                data_line.append(dictionary[(sen_cur[i] + '_' + sen_cur[i+1])])
                i = i + 2
            elif sen_cur[i] in dictionary:
                data_line.append(dictionary[(sen_cur[i])])
                i = i + 1
            elif len(sen_cur[i]) > 0:
                data_line.append(0)
                i = i + 1
        data.append(data_line)
    return data

### test on texts_to_sequences
sentences=[['i','take','this','seat','on','the','bus','today'], ['i','will','go','on','the','train']]
data = build_data_line(sentences, span=4)
dictionary = {}
dictionary["take_this_seat"]=1
dictionary["on_the"]=1
dictionary["take_this_seat"]=1
dictionary["on_the"]=2
dictionary["bus"]=3
dictionary["I"]=4
[[0, 1, 2, 3, 0], [0, 0, 0, 2, 0]]
### end of test

texts = [i[2] for i in sen]
sentences = []
for sen in texts:
    words = []
    for i in sen.split(' '):
        if len(i) > 0:
            words.append(i)
    sentences.append(words)

unigram = build_unigram(sentences)
bigram = build_bigram(sentences)
count = count_words(bigram, VOCAB_SIZE = 2000)
count = count_words(unigram, VOCAB_SIZE = 20000)
dictionary, reverse_dictionary = build_dictionary(count)
data = texts_to_sequences(sentences, span=4)
