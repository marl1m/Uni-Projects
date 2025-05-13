%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from collections import Counter
import nltk
from nltk.probability import FreqDist #nltk
from nltk.util import ngrams #nltk

from functions import *

import gc
from collections import Counter
import json



#################################### DATA DOWNLOAD #############################
data = pd.read_csv('data/train_preproc.csv', index_col= 'id').drop(columns='Unnamed: 0')
test_data = pd.read_csv('data/test_preproc.csv', index_col= 'id').drop(columns='Unnamed: 0')

data = data.loc[:,['lyrics_lemma', 'tag']].rename(columns={'lyrics_lemma': 'lyrics'}).copy()
test_data = test_data.loc[:,['lyrics_lemma']].rename(columns={'lyrics_lemma': 'lyrics'}).copy()


data.info()
data.head(3)
data[data.isna().any(axis=1)]
test_data[test_data.isna().any(axis=1)]


# data.drop(384771, inplace=True)
# data.drop(845262, inplace=True)


###################### 1st PREPARING BATCH ###########################
most_common_lyrics, fdist_percentages, tags, top_words = prep_data(data, 'lyrics','tag', n = 5000)

# # Find common keys and their locations
# common_keys = find_common_keys(top_words)

# print(f"Common keys are: {common_keys}")
      
######################### 1st LOG RATIO ANALYSIS #################################

log_ratios, total_freq = log_ratio_analysis(top_words, tags)

# Calculate the number of words for each genre
num_words = {
    'pop': int(3000 * (0.413008)),
    'rap': int(3000 * (0.285879)),
    'rock': int(3000 * (0.187692)),
    'rb': int(3000 * (0.045960)),
    'misc': int(3000 * (0.041781)),
    'country': int(3000 * (0.025681))
}

logs = {}
filtered_logs = {}
empty_keys = []

for genre in data['tag'].value_counts().keys():
    logs[genre] = sorted(log_ratios[genre].items(), key = lambda x: x[1], reverse=True)
    # Select the top N words for each genre
    filtered_logs[genre] = logs[genre][:num_words[genre]]
    # Check if the list is empty
    if not filtered_logs[genre]:
        empty_keys.append(genre)

print("Keys with empty lists:", empty_keys)

sum = 0
for genre in data.tag.value_counts().keys():
    print(f"{genre}: {len(filtered_logs[genre])} \n")
    sum += len(filtered_logs[genre])
print(sum)


'''    # sorted_few_logs[genre] = dict(list(log_ratios[genre].items())[:400])
    # print(f"{sorted_few_logs[genre]}\n")
    
common_keys = find_common_keys(sorted_few_logs)

key_frequencies = {key: len(locations) for key, locations in common_keys.items()}
sorted_key_frequencies = dict(sorted(key_frequencies.items(), key=lambda item: item[1], reverse=True))

for key, frequency in sorted_key_frequencies.items():
    # Only consider keys with a frequency greater than 2
    if frequency > 2:
        # Remove occurrences of the key from the 'lyrics' column
        # data['lyrics'] = data['lyrics'].apply(lambda x: (re.sub(rf'\b{key}\b', '', x)))
        data['lyrics'] = data['lyrics'].str.replace(f'\\b{key}\\b', '', regex=True)

        
        # Check if the key is still present in the 'lyrics' column
        if not data['lyrics'].str.contains(key).any():
            print(f'There is not a single "{key}" left in lyrics.')

########################### 2nd PREPARING BATCH ###############################
most_common_lyrics, fdist_percentages, tags, top_words = prep_data(data, 'lyrics','tag', n = 500)

# Find common keys and their locations
common_keys = find_common_keys(top_words)

print(f"Common keys are: {common_keys}")

key_frequencies = {key: len(locations) for key, locations in common_keys.items()}
sorted_key_frequencies = dict(sorted(key_frequencies.items(), key=lambda item: item[1], reverse=True))

for key, frequency in sorted_key_frequencies.items():
    # Only consider keys with a frequency greater than 2
    if frequency > 2:
        # Remove occurrences of the key from the 'lyrics' column
        # data['lyrics'] = data['lyrics'].apply(lambda x: (re.sub(rf'\b{key}\b', '', x)))
        data['lyrics'] = data['lyrics'].str.replace(f'\\b{key}\\b', '', regex=True)

        
        # Check if the key is still present in the 'lyrics' column
        if not data['lyrics'].str.contains(key).any():
            print(f'There is not a single "{key}" left in lyrics.')
            
########################## 2nd LOG RATIO ANALYSIS #############################
log_ratios2 = log_ratio_analysis(top20_2, tags2)

logs = {}
sorted_few_logs = {}
for genre in data['tag'].value_counts().keys():
    logs[genre] = sorted(log_ratios2[genre].items(), key = lambda x: x[1])
    print(logs[genre])

    sorted_few_logs[genre] = dict(list(log_ratios2[genre].items()))
    print(f"{sorted_few_logs[genre]}\n")

########################### 3rd PREPARING BATCH ###############################
most_common_lyrics, fdist_percentages, tags, top_words = prep_data(data, 'lyrics','tag', n = 500)

# Find common keys and their locations
common_keys = find_common_keys(top_words)

print(f"Common keys are: {common_keys}")

key_frequencies = {key: len(locations) for key, locations in common_keys.items()}
sorted_key_frequencies = dict(sorted(key_frequencies.items(), key=lambda item: item[1], reverse=True))

total_overlap = len(sorted_key_frequencies.keys())
overlap_more_2 = len([word for word, freq in sorted_key_frequencies.items() if freq > 2])
overlap_more_2 / total_overlap

for key, frequency in sorted_key_frequencies.items():
    # Only consider keys with a frequency greater than 2
    if frequency > 2:
        # Remove occurrences of the key from the 'lyrics' column
        # data['lyrics'] = data['lyrics'].apply(lambda x: (re.sub(rf'\b{key}\b', '', x)))
        data['lyrics'] = data['lyrics'].str.replace(f'\\b{key}\\b', '', regex=True)

        
        # Check if the key is still present in the 'lyrics' column
        if not data['lyrics'].str.contains(key).any():
            print(f'There is not a single "{key}" left in lyrics.')

########################### 4th PREPARING BATCH ###############################
most_common_lyrics, fdist_percentages, tags, top_words = prep_data(data, 'lyrics','tag', n = 500)

# Find common keys and their locations
common_keys = find_common_keys(top_words)

print(f"Common keys are: {common_keys}")

key_frequencies = {key: len(locations) for key, locations in common_keys.items()}
sorted_key_frequencies = dict(sorted(key_frequencies.items(), key=lambda item: item[1], reverse=True))

total_overlap = len(sorted_key_frequencies.keys())
overlap_more_2 = len([word for word, freq in sorted_key_frequencies.items() if freq > 2])
overlap_more_2 / total_overlap

for key, frequency in sorted_key_frequencies.items():
    # Only consider keys with a frequency greater than 2
    if frequency > 2:
        # Remove occurrences of the key from the 'lyrics' column
        # data['lyrics'] = data['lyrics'].apply(lambda x: (re.sub(rf'\b{key}\b', '', x)))
        data['lyrics'] = data['lyrics'].str.replace(f'\\b{key}\\b', '', regex=True)

        
        # Check if the key is still present in the 'lyrics' column
        if not data['lyrics'].str.contains(key).any():
            print(f'There is not a single "{key}" left in lyrics.')

########################## 3rd LOG RATIO ANALYSIS #############################
log_ratios3 = log_ratio_analysis(top20_3, tags3)

logs3 = {}
for genre in data['tag'].value_counts().keys():
    logs3[genre] = sorted(log_ratios3[genre].items(), key = lambda x: x[1])
    print(logs3[genre])
#####################################################################
'''


####################### MOST IMPORTANT WORDS FOR VECTORIZATION ###############

best_words = []
for tag in filtered_logs:
    for word in filtered_logs[tag]:
        best_words.append(word[0])

print(best_words)
len(set(best_words)) #3496 palavras distintas


# verifying which words are repeating

counter = Counter(best_words)

repeated_words = [item for item, count in counter.items() if count > 1]

if repeated_words:
    print("The list contains these repeated values:", repeated_words)
else:
    print("The list does not contain any repeated values.")



##############CHECK FOR WORSD NOT IN LYRICS OF TEST SET ###################

# all lyrics from test
all_words = ' '.join([word for word in test_data['lyrics']])   
tokenized_words = nltk.tokenize.word_tokenize(all_words)   
len(tokenized_words) #4552683

#all words from the vocabulary not in test_set lyrics
not_in = []
for word in best_words:
    if word not in all_words:
        not_in.append(word)
print(len(not_in))  

#remove those words from vocabulary
for word in not_in:
    best_words.remove(word)
    
'''# mask = data['lyrics'].str.contains('primeira', regex=True)

# Apply the mask to the DataFrame to get the rows where 'sentence' appears
# result = data[mask].lyrics

# data.lyrics.loc[1068883]
# data.lyrics.loc[2223982]'''
###################### LISTA FINAL AINDA NAO ESTA DECIDIDA !!!! ##########


# Assuming 'all_keys' is your list
with open('vocabulary7.py', 'w') as file:
    file.write('vocabulary = ' + json.dumps(best_words))
    
    ################################################################