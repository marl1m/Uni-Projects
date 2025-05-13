%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
import re
import nltk 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag_sents

# Ensure NLTK data is downloaded
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('punkt')

from functions import *

############################# DOWNLOAD DATA ############################
df = pd.read_csv('data/train.csv', index_col = 'id')
df_test = pd.read_csv('data/test.csv', index_col = 'id')


############################ LOWERIZATION ##############################
df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).apply(lambda x: x.str.lower())
df_test[df_test.select_dtypes(['object']).columns] = df_test.select_dtypes(['object']).apply(lambda x: x.str.lower())

######################### CONTRACTIONS AND ABREVIATIONS #############################
contractions = {
"verse": "",
"chorus": "",
"i ain't": "i am not",
"he ain't": "he is not",
"she ain't": "she is not", 
"it ain't": "it is not", 
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"gimme" : "give me",
"gotta" : "have got",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he shall",
"he'll've": "he shall have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "I shall",
"i'll've": "i shall have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it shall",
"it'll've": "it shall have ",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she shall",
"she'll've": "she shall have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as ",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": " they would",
"they'd've": "they would have",
"they'll": "they shall ",
"they'll've": "they shall have ",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": " we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall ",
"what'll've": "what shall have ",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who shall",
"who'll've": "who shall have",
"who's": " who is",
"who've": "who have",
"why's": " why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you shall ",
"you'll've": "you shall have",
"you're": "you are",
"you've": "you have",
"youve" : "you have" 
}

df.lyrics.replace(contractions, regex=True, inplace = True)
df_test.lyrics.replace(contractions, regex=True, inplace = True)

######################### RANDOM STUFF REMOVAL IN LYRICS ####################
df['lyrics_clean'] = df['lyrics'].apply(lambda x: remove(x))
df_test['lyrics_clean'] = df_test['lyrics'].apply(lambda x: remove(x))

########################### TOKENIZATION #################################
df['lyrics_tokened'] = df['lyrics_clean'].apply(tokenize_text)
df_test['lyrics_tokened'] = df_test['lyrics_clean'].apply(tokenize_text)

########################## LANGUAGE DETECTION #########################
df['language'] = df['lyrics_tokened'].apply(language_detector)
non_english_rows = df[df['language'] != 'en']
df.drop(non_english_rows.index, inplace = True)

df_test['language'] = df_test['lyrics_tokened'].apply(language_detector)
non_english_rows = df_test[df_test['language'] != 'en']

########################## STOPWORD REMOVAL ###############################
stopwords = nltk.corpus.stopwords.words("english")

df['lyrics_tokened_filtered'] = df['lyrics_tokened'].apply(
    lambda x: [item for item in x if item not in stopwords])


df['lyrics_string_clean'] = df['lyrics_tokened_filtered'].apply(lambda x: ' '.join(
                                                            [word for word in x if len(word) > 3]))

df_test['lyrics_tokened_filtered'] = df_test['lyrics_tokened'].apply(
    lambda x: [item for item in x if item not in stopwords])


df_test['lyrics_string_clean'] = df_test['lyrics_tokened_filtered'].apply(lambda x: ' '.join(
                                                            [word for word in x if len(word) > 3]))

############################ LEMMATIZATION #############################################################################
df['POSTags'] = pos_tag_sents(df['lyrics_string_clean'].apply(word_tokenize).tolist())
df_test['POSTags'] = pos_tag_sents(df_test['lyrics_string_clean'].apply(word_tokenize).tolist())

def lemmatize_word(word, pos):
    pos_dict = {'J':wordnet.ADJ, 'V':wordnet.VERB, 'N':wordnet.NOUN, 'R':wordnet.ADV}
    pos = pos_dict.get(pos[0], wordnet.NOUN)
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word, pos)

df['lyrics_lemma'] = df['POSTags'].apply(lambda x: [lemmatize_word(y, z) for y, z in x])
df_test['lyrics_lemma'] = df_test['POSTags'].apply(lambda x: [lemmatize_word(y, z) for y, z in x])

################################ GET THE FINAL PROCESSED DATA #########################

df['lyrics_lemma'] = df['lyrics_lemma'].apply(
                                              lambda x: ' '.join(
                                              [item for item in x ]))

df_test['lyrics_lemma'] = df_test['lyrics_lemma'].apply(
                                              lambda x: ' '.join(
                                              [item for item in x ]))
pre_proc_train = df[['year','views',
                         'title',
                         'tag',
                         'artist',
                         'features',
                         'lyrics_lemma']].reset_index()

pre_proc_test = df_test[['year','views',
                         'title',
                          'artist',
                         'features',
                         'lyrics_lemma']].reset_index()

pre_proc_train.to_csv('data/train_preproc.csv', index= 'id')
pre_proc_test.to_csv('data/test_preproc.csv', index= 'id')
