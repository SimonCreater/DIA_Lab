import tarfile
import pyprind
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()

def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:]
if w not in stop]
count=CountVectorizer()
# with tarfile.open('aclImdb_v1.tar.gz','r:gz') as tar:
#   tar.extractall()
# basepath = 'aclImdb'

# labels = {'pos': 1, 'neg': 0}
# pbar = pyprind.ProgBar(50000)

# for s in ('test', 'train'):
#     for l in ('pos', 'neg'):
#         path = os.path.join(basepath, s, l)
#         for file in sorted(os.listdir(path)):
#             with open(os.path.join(path, file), 
#                       'r', encoding='utf-8') as infile:
#                 txt = infile.read()
#             df = df.append([[txt, labels[l]]], 
#                            ignore_index=True)
#             pbar.update()


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
lr_tfidf=Pipeline([('vect',tfidf),('clf',LogisticRegression(solver='liblinear',random_state=0))])
df_train=pd.read_csv('raiting_train.txt',decimal='\t',keep_default_na=False)
X_train=df_train['document'].values
y_train=df_train['label'].values

df_test=pd.read_csv('raiting_test.txt',decimal='\t',keep_default_na=False)
X_train=df_test['document'].values
y_train=df_test['label'].values

