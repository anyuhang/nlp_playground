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

def readSearchTerms(fn):
  sen = []
  for sentence in read_sentences(open(fn)):
    a = sentence.split(',')
    if len(a) == 7:
        sen.append([a[1], int(a[6])])
    elif len(a) > 2:
        sen.append([a[1], 1])
  return sen

top_word_index={w:i for i,w in index_word.items()[0:20000]}
fn_gs = '/Users/yan/Code/avvo_nlp/search_terms-201703.csv'
search = readSearchTerms(fn_gs)
wordCnt = []
for i in xrange(len(search)):
    cnt = 0
    for w in search[i][0].split(' '):
        if w in top_word_index:
            cnt += 1
    if cnt > 0:
        wordCnt.append([i, cnt])
