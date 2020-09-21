from gensim.models import Word2Vec
import re
import sys
import csv
from nltk.tag import pos_tag
#from nltk.stem.snowball import SnowballStemmer
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

from nltk.corpus import stopwords 
all_stopwords = stopwords.words('english')

#use word stems
#stemmer = SnowballStemmer("english")

if len(sys.argv)==4:

    #paths with read data
    datapath=str(sys.argv[1])
    modelpath=str(sys.argv[2])
    wordstopredict=str(sys.argv[3])



    #open file and read data
    if not datapath.endswith(".csv"):
        datapath=datapath+str(".csv")

    #save as .bin
    if not modelpath.endswith(".bin"):
        modelpath=modelpath+str(".bin")

    f = open(datapath, newline='', encoding='utf-8')
    reader = csv.reader(f, delimiter=',')
    #tags=f.readlines()
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


else:
    #paths without read data
    modelpath=str(sys.argv[1])
    wordstopredict=str(sys.argv[2])

    #save as .bin
    if not modelpath.endswith(".bin"):
        modelpath=modelpath+str(".bin")


f=open(wordstopredict)
predictwords=[]
i=0
l=f.readline()
#read data
while(l != ''):
    a=l.split()
    l2=[]
    for j in range(0,len(a),1):
        l2.append(str(a[j].lower()))
    l2=l2[len(a)-1]
    predictwords.append(l2)
    l=f.readline()


#stem the preditwords
#predictwords = [stemmer.stem(w) for w in predictwords]


model = Word2Vec.load(modelpath)
for i in range(len(predictwords)):
    similarwords=model.wv.most_similar(positive=[predictwords[i]], topn=5)
    print(predictwords[i],' : ',similarwords[0][0],' , ',similarwords[1][0],' , ',similarwords[2][0],' , ',similarwords[3][0],' , ',similarwords[4][0])


