from gensim.models import Word2Vec
import re
import sys
import csv
from nltk.tag import pos_tag
#from nltk.stem.snowball import SnowballStemmer
import nltk
from nltk.tokenize import RegexpTokenizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

from nltk.corpus import stopwords 
all_stopwords = stopwords.words('english')

#paths
datapath=str(sys.argv[1])
modelpath=str(sys.argv[2])

size=300
window=2
realorfake=0
'''
size=int(sys.argv[3])
window=int(sys.argv[4])
realorfake=int(sys.argv[5])
#!!!!!!!!!!!!!!!!!!!!!!REMOVE DATA FILTERING
'''
print('size:',size)
print('window:',window)



#use word stems
#stemmer = SnowballStemmer("english")

#Tokenize from professors code
def tokenizedata(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
            if re.search('[a-zA-Z]', token):
                    filtered_tokens.append(token)
    #filtered_tokens = [stemmer.stem(t) for t in filtered_tokens]
    return filtered_tokens


#open file and read data and check the extensions
if not datapath.endswith(".csv"):
    datapath=datapath+str(".csv")

#save as .bin
if not modelpath.endswith(".bin"):
    modelpath=modelpath+str(".bin")

f = open(datapath, newline='', encoding='utf-8')
reader = csv.reader(f, delimiter=',')
data = []
labels=[]
i = 0
toremove = {}
for row in reader:
        #remove non-ascii from the line
        if(i == 0):
            i = 1
            continue
        line = row[2]
        line = re.sub(r"[^\x00-\x7F]+", "", line)
        line = re.sub(r"\n", "", line)
        if(row[3] == 'FAKE'):
                labels.append(1)
        else:
                labels.append(0)
        data.append(line)
        
#print(len(data))
#print(len(labels))
#print('running short data')
#data=data[0:110]
#labels=labels[0:110]
'''
#Change datasets to only real or fake
newdata=[]
for i in range(len(data)):
        if labels[i]==int(sys.argv[5]):
                newdata.append(data[i])
data=newdata
'''
#remove punctuation
tokenizertoremovepunct = RegexpTokenizer(r'\w+')
for i in range(0, len(data), 1):
        tok = tokenizertoremovepunct.tokenize(data[i])
        newstr = ""
        for t in tok:
                newstr += t+" "
        data[i] = newstr

#Clean data from professros code
for i in range( len(data)):
	tagged = pos_tag(data[i].split())
	toremove = [word for word,pos in tagged if( pos == 'CC' or pos == 'CD' or pos == 'DT' or pos == 'EX' or \
							pos == 'IN' or pos == 'LS' or pos == 'MD' or pos == 'DT' or \
						    	pos == 'PRP' or pos == 'PRP$' or pos == 'RP' or pos == 'TO' or \
							pos == 'UH' or pos == 'WDT' or pos == 'WP' or pos == 'WP$' ) ]

	tok = tokenizedata(data[i])
        #remove stop words
	tok=[t for t in tok if not t in all_stopwords]
	newstr = ""
	for t in tok:
		flag=0
		for j in range(0, len(toremove), 1):
			if(toremove[j] == t):
				flag=1
		if(flag == 0):
			newstr += t+" "
	data[i] = newstr.lower()

#some final cleaning - remove single letters(Professors code)
for i in range(0, len(data), 1):
        tok = tokenizedata(data[i])
        newstr = ""
        for t in tok:
                if(len(t) != 1):
                        newstr += t+" "
        data[i] = newstr.lower()
        data[i]  = tokenizedata(data[i])



model = Word2Vec(data,size=size, window=window )

model.save(modelpath)

#print('done')
