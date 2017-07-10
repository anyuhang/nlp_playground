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
def count_words(count, words, VOCAB_SIZE=20000):
    import collections
    count.extend(collections.Counter(words).most_common(VOCAB_SIZE - 1))
    return count

def build_dictionary(count):
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return  dictionary, reverse_dictionary

# step 3: get sequence of data_line using dictionary
def texts_to_sequences(sentences, dictionary, span=4):
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

## generate multi_gram from topic keywords
def readKeyWords(fn):
    import re
    sen = []
    for sentence in read_sentences(open(fn)):
          sen_clean = re.sub(r"specialty:\'.*\'", '', sentence).strip()
          sen_clean = re.sub(r"specialty:.*", '', sen_clean).strip()
          sen_clean = re.sub(r"-tag:.*", '', sen_clean).strip()
          sen_clean = re.sub(r"tag:.*", '', sen_clean).strip()
          sen_clean = re.sub(r"tag:\".*\"", '', sen_clean).strip()
          sen_clean = re.sub(r"\(.*\)", '', sen_clean).strip()
          sen_clean = re.sub(r'^"|"$', '', sen_clean).strip()
          sen_clean = re.sub(r'^\\|\\$', '', sen_clean).strip()
          sen_clean = re.sub(r'^"|"$', '', sen_clean).strip()
          sen_clean = re.sub(r'^\\|\\$', '', sen_clean).strip()
          sen_clean = re.sub(r'\\"$', '', sen_clean).strip()
          sen_clean = re.sub(r'^\'|\'$', '', sen_clean).strip()
          sen_clean = re.sub(r'^"$', '', sen_clean).strip()
          sen_clean = re.sub(r'^\\$', '', sen_clean).strip()
          sen_clean = re.sub(r'\\', '', sen_clean).strip()
          sen_clean = re.sub(r'\\', '', sen_clean).strip()
          sen_clean = re.sub(r'\"', '', sen_clean).strip()
          sen_clean = re.sub(r'\"', '', sen_clean).strip()
          if len(sen_clean) > 0:
              sen.append(sen_clean)
    return sen

def get_kw_list(kw_file):
    keywords = readKeyWords(kw_file)
    kw_dedup = list(set(keywords))
    kw_list = []
    for i in kw_dedup:
        gram_cnt = len(i.split(' '))
        kw_list.append([i.lower().replace(' ', '_'), gram_cnt])
    return kw_list
