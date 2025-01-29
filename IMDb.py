import tarfile
import pyprind
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import nltk
nltk.download('stopwords')
porter = PorterStemmer()

def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:]
if w not in stop]
count=CountVectorizer()

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text

df=pd.read_csv('movie_data.csv',encoding='utf-8')
df.head(3)
docs = np.array([
        'The sun is shining',
        'The weather is sweet',
        'The sun is shining, the weather is sweet, and one and one is two'])

bag = count.fit_transform(docs)
print(count.vocabulary_)
print(bag.toarray())
tfidf=TfidfTransformer(use_idf=True,
                       norm='12',
                       smooth_idf=True)
# print(tfidf.fit_transform(count.fit_transform(docs)).toarray())
porter = PorterStemmer()
def tokenizer_porter(text):
  return [porter.stem(word) for word in text.split()]
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values
print(X_train)
tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)
param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              ]

#대용량 데이터 처리리
def stream_docs(path):#한 줄씩 읽으면서 text와 레이블 구별하는 역할
    with open(path,'r',encoding='utf-8') as csv:
        next(csv)
        for line in csv:
            text,label=line[:-3],int(line[-2])
            yield text,label



def get_minibatch(doc_stream,size):
    docs,y=[],[]
    try:
        for _ in range(size):
            text,label=next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        pass
    return docs,y        


vect=HashingVectorizer(decode_error='ignore',#오류발생 무시
                       n_features=2**21,#해싱 벡터
                       preprocessor=None,
                       tokenizer=tokenizer)
clf=SGDClassifier(loss='log',random_state=1,max_iter=1)

doc_stream=stream_docs(path='movie_data.csv')

pbar=pyprind.ProgBar(45)
classes=np.array([0,1])

for _ in range (45):
    X_train,y_train=get_minibatch(doc_stream,size=1000)
    if not X_train:
        break
    X_train=vect.transform(X_train)
    clf.partial_fit(X_train,y_train,classes=classes)
    pbar.update()
#토픽 모델링

count=CountVectorizer(stop_words='english',
                      max_df=.1,
                      max_features=5000)
X=count.fit_transform(df['review'].values)
lda=LatentDirichletAllocation(n_components=10,
                              random_state=123,
                              learning_method='batch')
X_topics=lda.fit_transform(X)

n_top_words=5
feature_names=count.get_feature_names_out()
for topic_idx,topic in enumerate(lda.components_):
    print("토픽 %d" %(topic_idx+1))
    print(" ".join([feature_names[i] for i in 
                    topic.argsort()[:-n_top_words -1:-1]]))#가장 중요한 n_top_words개 단어만 선택 (역순으로 출력)
