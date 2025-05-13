%load_ext autoreload
%autoreload 2

from tqdm import tqdm
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from collections import Counter
from nltk.probability import FreqDist #nltk
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer #sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from nltk.util import ngrams #nltk

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

sns.set()

import gensim.downloader as api
from gensim.models import Word2Vec


from functions import *
# from list_export import all_keys #1978
# from vocabulary import vocabulary #2954

from vocabulary1 import vocabulary1 #1781 DIFERENTES!!!!!

# from vocabulary4 import vocabulary #2741 

# from vocabulary5 import vocabulary #3851
# from vocabulary6 import vocabulary #3496 with cap log_ratio 
# from vocabulary7 import vocabulary #2998

# join vocabularies
# new_vocabulary = list(set(vocabulary) & set(vocabulary1))


################################### DATA ##############################

data = pd.read_csv('data/train_preproc.csv', index_col= 'id').drop(columns='Unnamed: 0')
test_data = pd.read_csv('data/test_preproc.csv', index_col= 'id').drop(columns='Unnamed: 0').sort_index()
sample_submission = pd.read_csv('data/sample_submission.csv', index_col= 'id').sort_index()

data.info()
test_data.info()

data[data.isna().any(axis=1)]
test_data[test_data.isna().any(axis=1)]

# data.drop(384771, inplace=True)
# data.drop(845262, inplace=True)

data = data.rename(columns={'lyrics_lemma': 'lyrics'})
test_data = test_data.rename(columns={'lyrics_lemma': 'lyrics'})

###################### PLOTTING SOME MODELS TO COMPARE ############################### 
'''

models = [
    LinearSVC(penalty='l2',loss='squared_hinge',dual=False,tol=1e-4,C=1.0,multi_class='ovr',fit_intercept=True,intercept_scaling=1,class_weight='balanced',random_state=42),  
    MultinomialNB(),
    LogisticRegression(solver='saga', random_state=42, class_weight="balanced"), 
]

other_models = [
    GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42),
    ExtraTreesClassifier(n_estimators=100, max_depth=None, random_state=42),
    MLPClassifier(hidden_layer_sizes=(2,), random_state=42),
    XGBClassifier(use_label_encoder=True, eval_metric='mlogloss'),
    SVC(kernel='linear', C=1.0, random_state=42,class_weight='balanced'),
    SGDClassifier(loss='hinge', penalty='l2', random_state=42),
    KNeighborsClassifier(n_neighbors=5),
    DecisionTreeClassifier(max_depth=5, random_state=42)
]

# 5 Cross-validation
CV = 2
cv_df = pd.DataFrame(index=range(CV * len(other_models)))
entries = []
for model in other_models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='f1_weighted', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])


plt.figure(figsize=(8,5))
sns.boxplot(x='model_name', y='f1score', 
            data=cv_df, 
            color='lightblue', 
            showmeans=True)
plt.title("MEAN F1SCORE (cv = 1)", size=14)
'''


################################### PIPELINE ########################### 
'''# # CountVectorizer
# cv = CountVectorizer(vocabulary = set(vocabulary))

# train_features = cv.fit_transform(data.lyrics).toarray()
# train_labels = data.tag

# X_test = cv.transform(test_data.lyrics).toarray()
'''
# TDIF
tfidf = TfidfVectorizer(sublinear_tf=True,
                        min_df=5,
                        ngram_range=(1, 2), 
                        stop_words='english', 
                        vocabulary=set(vocabulary1)
                        )
#train the vectorizer
train_features = tfidf.fit_transform(data.lyrics).toarray()
train_labels = data.tag

X_test = tfidf.transform(test_data.lyrics).toarray()

# from sklearn.feature_selection import SelectKBest, chi2
# k_best = SelectKBest(chi2, k=1500)
# train_features = k_best.fit_transform(train_features, train_labels)

# X_test = k_best.transform(X_test)

#split data
X_train, X_val, y_train, y_val = train_test_split(train_features, 
                                                               train_labels, 
                                                               test_size=0.25, 
                                                               random_state=42)
#model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

#predicts
train_pred = model.predict(X_train)
val_pred = model.predict(X_val)

# Model Evaluation Function
metrics_pretty(y_train, train_pred, y_val, val_pred)

############################## SUBMISSION ####################
final_pred = model.predict(X_test)

sample_submission['tag'] = final_pred
sample_submission.to_csv('data/Group03_Version38.csv')

################################ GRID SEARCH #################################
               
'''from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression

# Define the parameter grid
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2']
}

# Initialize the Logistic Regression model
log_reg = LogisticRegression()

# Initialize the GridSearchCV
grid_search = RandomizedSearchCV(log_reg, param_grid, cv=2, scoring='f1_weighted', verbose=2)

# Fit the GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters #40min to run...
best_params = grid_search.best_params_
print(best_params)
best_score = grid_search.best_score_
print(best_score)'''

#########################################################################
param_grid = [
    {'penalty': ['l1'], 
     'solver': ['liblinear', 'saga'],
     'C': [0.001, 0.01, 0.1, 1, 10, 100,150,500],
     'max_iter': [100, 200, 300, 400, 500],
     'multi_class': ['auto', 'ovr', 'multinomial']},
    
    {'penalty': ['l2', None], 
     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
     'C': [0.001, 0.01, 0.1, 1, 10, 100,150,500],
     'max_iter': [100, 200, 300, 400, 500],
     'multi_class': ['auto', 'ovr', 'multinomial']},
]    
   

# Initialize the Logistic Regression model
log_reg = LogisticRegression()

# Initialize the GridSearchCV
grid_search = RandomizedSearchCV(log_reg, param_grid, cv=2, scoring='f1_weighted', verbose=2)

# Fit the GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params2 = grid_search.best_params_
print(best_params2)
best_score2 = grid_search.best_score_
print(best_score2)



'''################################## HIPER CLASS MODELO #############################

from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.layers import Layer, Input, Embedding, LSTM, Dense, TimeDistributed, Bidirectional
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import Callback
from sklearn.metrics import f1_score

class Attention(Layer):
    
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)


class F1Score(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict, average='micro')
        self.val_f1s.append(_val_f1)
        print(f' — val_f1: {_val_f1}')

# Parameters
MAX_SENT_LENGTH = 100
MAX_SENTS = 15
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# Model architecture
sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedded_sequences = Embedding(MAX_NB_WORDS, EMBEDDING_DIM)(sentence_input)
l_lstm = Bidirectional(LSTM(100, return_sequences=True))(embedded_sequences)
l_dense = TimeDistributed(Dense(200))(l_lstm)
l_att = Attention(MAX_SENT_LENGTH)(l_dense)
sentEncoder = Model(sentence_input, l_att)

review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(LSTM(100, return_sequences=True))(review_encoder)
l_dense_sent = TimeDistributed(Dense(200))(l_lstm_sent)
l_att_sent = Attention(MAX_SENTS)(l_dense_sent)
preds = Dense(5, activation='softmax')(l_att_sent)
model = Model(review_input, preds)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['f1score'])

# Fit the model
f1score = F1Score()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=50, callbacks=[f1score])
'''

#word2vec
'''from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# WORD 2 VEC

lyrics_list = data['lyrics'].tolist()

tokenized_lyrics = [word_tokenize(lyric.lower()) for lyric in lyrics_list]

model = Word2Vec(sentences=tokenized_lyrics,
                 vector_size = int(np.mean([len(sentence) for sentence in tokenized_lyrics])),
                 window=10,
                 min_count=3)

model.train(tokenized_lyrics, total_words = len(tokenized_lyrics), epochs=200)

class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = 10

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                or [np.zeros(self.dim)], axis=0)
            for words in X
    ])


word2vec = Word2VecVectorizer(model.wv)

from sklearn.svm import SVC

pipeline = Pipeline([
    ("word2vec vectorizer", word2vec),
    ("logistic regression", LogisticRegression(random_state=42))
])


pipeline.fit(X_train, y_train)

train_pred = pipeline.predict(X_train)
val_pred = pipeline.predict(X_val)

metrics_pretty(y_train, train_pred, y_val, val_pred)

final_pred = pipeline.predict(X_test)

sample_submission['tag'] = final_pred
sample_submission.to_csv('data/Group03_Version30.csv')'''











