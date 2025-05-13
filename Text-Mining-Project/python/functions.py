import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer

# from nltk.tag import pos_tag
from nltk.tag import pos_tag_sents
from nltk.tokenize import word_tokenize

stopwords = nltk.corpus.stopwords.words("english")
import matplotlib.pyplot as plt
import plotly.express as px
from typing import Optional
from wordcloud import WordCloud

import seaborn as sns
sns.set()
from collections import Counter
import nltk
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
#word2vec


from sklearn.model_selection import train_test_split

import gensim
import gensim.downloader as api
from gensim.models import Word2Vec
from langdetect import detect

import nltk
nltk.download('universal_tagset')
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('wordnet')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from textblob import TextBlob
from flair.data import Sentence
from flair.nn import Classifier


###################### VISUALIZATION #########################

#### Line Plot Function ####
def create_line_plot(data: pd.DataFrame, x_column: str, y_column: str, 
                     title: str = '', xlabel: str = '', ylabel: str = ''):
    """
    Create a line plot to display variation of a variable over time.

    Parameters:
        data (DataFrame): The input DataFrame.
        x_column (str): The column to be used on the x-axis (time column).
        y_column (str): The column to be used on the y-axis (variable to be represented).
        title (str): The title of the plot (default is an empty string).
        xlabel (str): The label for the x-axis (default is an empty string).
        ylabel (str): The label for the y-axis (default is an empty string).
    """
    plt.figure(figsize=(12, 6))
    
    sns.lineplot(x=x_column, y=y_column, data=data, ci=None, color='#0c3383', label=y_column)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

#### Donut Chart ####
def create_donut_chart(data: pd.DataFrame, value_column: str, names_column: str,
                       title: str = '', labels: dict = None):
    """
    Create a donut chart using Plotly.

    Parameters:
        data (DataFrame): The input DataFrame.
        value_column (str): The column containing values for the chart.
        names_column (str): The column containing names for the chart.
        hole (float): The size of the center hole in the donut (default is 0.4).
        title (str): The title of the plot (default is an empty string).
        labels (dict): A dictionary specifying labels for columns (default is None).
        template (str): The Plotly template for the chart (default is 'plotly_dark').
    """
    tag_counts = data[names_column].value_counts().reset_index()

    fig = px.pie(tag_counts, values=value_column, names=names_column, hole=0.4,
                 labels=labels, title=title)

    # Update trace for better visibility
    fig.update_traces(textposition='inside',
                      textinfo='percent+label',
                      marker=dict(colors=px.colors.diverging.Portland))

    # Show the plot
    fig.show()

#### Word Frequency Chart ####
def bar_chart_builder(x, y, title):
    plt.figure(figsize=(10, 6))
    plt.barh(x, y, color='#0c3383')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(False)  # Add this line to remove the grid
    plt.gca().set_facecolor('white')  # Add this line to remove the gray background
    plt.show()

def frequency_word_chart(data: pd.DataFrame, filtered_string: str):
    """
    Create bar chart to compare the frequency of various words.

    Parameters:
        data (DataFrame): The input DataFrame.
        filtered_string (string): The name of the column belonging to the dataframe whose terms we will be comparing.
    """
    all_words = ' '.join([word for word in data[filtered_string]])
    tokenized_words = nltk.tokenize.word_tokenize(all_words)
    fdist = FreqDist(tokenized_words)
    first_15_items = fdist.most_common(15)
    words = [item[0] for item in first_15_items]
    words_freq = [item[1] for item in first_15_items]
    bar_chart_builder(x=words, y=words_freq, title=filtered_string.split('_')[0].upper())

#### Wordcloud ####
def wordcloud_generator(sampled_lyrics: str, title: str, title_fontsize: Optional[int] = 16) -> None:
    """
    Generate a word cloud and display it using matplotlib.

    Parameters:
    - sampled_lyrics (str): The sampled lyrics data for generating the word cloud.
    - title (str): The title for the wordcloud plot.
    - title_fontsize (int, optional): The font size for the title. The default value is 16.

    Returns:
    None
    """
    # Generate a word cloud for the sampled data
    wordcloud = WordCloud(width=800, height=400, max_words=200, background_color='white').generate(sampled_lyrics)

    # Display the generated word cloud using matplotlib
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontdict={'fontsize': title_fontsize})  # Set the font size for the title
    plt.axis('off')
    plt.show()

#### Wordcloud For Tags####
def wordcloud_generator_for_tags(data, tag_column, lyrics_column, title_fontsize: Optional[int] = 16) -> None:
    """
    Generate word clouds for each unique tag value and display them as subplots in a single figure using matplotlib.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the data.
    - tag_column (str): The column containing tag values.
    - lyrics_column (str): The column containing lyrics data.
    - title_fontsize (int, optional): The font size for the title. The default value is 16.

    Returns:
    None
    """
    unique_tags = data[tag_column].unique()
    num_tags = len(unique_tags)
    rows = num_tags // 2 + num_tags % 2  # Calculate the number of rows for subplots

    # Create a subplot grid
    fig, axes = plt.subplots(rows, 2, figsize=(12, 4 * rows))

    for i, current_tag in enumerate(unique_tags):
        # Determine the current subplot position
        row = i // 2
        col = i % 2

        # Filter data for the current tag
        tag_lyrics = ' '.join(data[data[tag_column] == current_tag][lyrics_column].dropna().sample(n=1000))

        # Generate a word cloud for the current tag
        wordcloud = WordCloud(width=400, height=200, max_words=100, background_color='white').generate(tag_lyrics)

        # Display the generated word cloud as a subplot
        axes[row, col].imshow(wordcloud, interpolation='bilinear')
        axes[row, col].set_title(f'Word Cloud for Tag: {current_tag}', fontsize=title_fontsize)
        axes[row, col].axis('off')

    # Adjust layout and show the plot with reduced vertical space between rows
    plt.subplots_adjust(hspace=0.3)
    plt.tight_layout()
    plt.show()

def wordcloud_generator_for_emotions(data, emotion_column, lyrics_column, title_fontsize: Optional[int] = 16, sample_size: Optional[int] = 1000) -> None:
    """
    Generate word clouds for each unique emotion value and display them as subplots using matplotlib.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the data.
    - emotion_column (str): The column containing emotion values.
    - lyrics_column (str): The column containing lyrics data.
    - title_fontsize (int, optional): The font size for the title. The default value is 16.
    - sample_size (int, optional): The number of samples to use when creating word clouds. The default value is 1000.

    Returns:
    None
    """
    unique_emotions = data[emotion_column].unique()
    num_emotions = len(unique_emotions)
    rows = num_emotions // 3 + num_emotions % 3  # Calculate the number of rows for subplots

    # Create a subplot grid
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))

    # Determine the color palette for emotions
    palette = sns.color_palette("Dark2", n_colors=num_emotions)

    for i, emotion in enumerate(unique_emotions):
        # Determine the current subplot position
        row = i // 3
        col = i % 3

        # Filter data for the current emotion
        subset_data = data[data[emotion_column] == emotion]
        
        # Check if sample_size is greater than the number of available data points
        actual_sample_size = min(sample_size, len(subset_data))
        
        # Combine the lyrics for the current emotion and use a sample
        wordcloud_input = ' '.join(subset_data[lyrics_column].dropna().sample(n=actual_sample_size))
        
        # Generate WordCloud
        wordcloud = WordCloud(width=400, height=200, max_words=200, background_color='white').generate(wordcloud_input)

        # Display the generated word cloud as a subplot
        axes[row, col].imshow(wordcloud, interpolation='bilinear')
        axes[row, col].set_title(f'Word Cloud for Emotion: {emotion}', fontsize=title_fontsize)
        axes[row, col].axis('off')

    # Adjust layout and show the plot with reduced vertical space between rows
    plt.subplots_adjust(hspace=0.3)
    plt.tight_layout()
    plt.show()

def wordcloud_generator_for_sentiments(data, sentiment_column, lyrics_column, title_fontsize: int = 16):
    """
    Generate and display word clouds for positive and negative sentiment using matplotlib.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the data.
    - sentiment_column (str): The column containing sentiment values (e.g., 'vader_song').
    - lyrics_column (str): The column containing lyrics data.
    - title_fontsize (int, optional): The font size for the title. The default value is 16.

    Returns:
    None
    """
    # Filter data for positive and negative sentiment
    positive_sentiment_data = data[data[sentiment_column] > 0]
    negative_sentiment_data = data[data[sentiment_column] < 0]

    # Combine the lyrics for positive and negative sentiment
    positive_wordcloud_input = ' '.join(positive_sentiment_data[lyrics_column])
    negative_wordcloud_input = ' '.join(negative_sentiment_data[lyrics_column])

    # Use the 'Dark2' colormap for both positive and negative sentiment
    colormap = 'Dark2'

    # Generate WordCloud for positive sentiment
    positive_wordcloud = WordCloud(width=800, height=600,
                                   background_color='white',
                                   max_font_size=50,
                                   colormap=colormap).generate(positive_wordcloud_input)

    # Generate WordCloud for negative sentiment
    negative_wordcloud = WordCloud(width=800, height=600,
                                   background_color='white',
                                   max_font_size=50,
                                   colormap=colormap).generate(negative_wordcloud_input)

    # Plot the WordCloud image for positive sentiment
    plt.figure(figsize=(8, 6), facecolor=None)
    plt.imshow(positive_wordcloud, interpolation='bilinear')
    plt.title('Word Cloud for Positive Sentiment', fontsize=title_fontsize)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    # Plot the WordCloud image for negative sentiment
    plt.figure(figsize=(8, 6), facecolor=None)
    plt.imshow(negative_wordcloud, interpolation='bilinear')
    plt.title('Word Cloud for Negative Sentiment', fontsize=title_fontsize)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

###################### PREPROCESSING #########################

def remove(text):
    # This function will remove regex expressions in the text, alongside emojis and anything within []
    # Remove emojis
    filtered_text = text.encode('ascii', 'ignore').decode('ascii')
    # Replace expressions such as \n, \t, \r, \f and \v with a space
    filtered_text = re.sub(r'\\[ntrfv]', ' ', repr(filtered_text))
    # Replace isolated consonants with a space:
    filtered_text = re.sub(r'\b([^aeiou])\b',' ',filtered_text)
    # Remove regex expressions - [^0-9A-Za-z'\\ \t]
    filtered_text = re.sub(r"\[|\]|\(|\)| @\[A-Za-z0-9]+|\d+|[^0-9A-Za-z \t]|\w+:\/\/\S+|^rt|http.+?", ' ', filtered_text)
    # Remove extra spaces
    filtered_text = re.sub(r' +', ' ', filtered_text)
       
    return filtered_text.lower()

# Function to detect language of a list of words
 #For translating the data
def language_detector(words):
    """
    Function that will analyse all words within a text sample to identify existence of terms 
    from different language.

    Parameters:
    words (list): the list of words that are to be analyzed one by one     
    """
    try:
        # Joining the words into a string and detecting the language
        lang = detect(' '.join(words))
        return lang
    except:
        # If language detection fails, return 'unknown'
        return 'unknown'

# Function to filter rows based on any word count
def filter_rows_by_any_word_count(dataframe, count_threshold):
    # Count occurrences of each word in each row
    word_counts = dataframe['lyrics_tokenized'].apply(lambda x: pd.Series(x).value_counts()).sum(axis=1)

    # Filter rows where the word count is greater than the threshold
    filtered_rows = dataframe[word_counts > count_threshold]

    return filtered_rows

def tokenize_text(text):
    if text is not None and isinstance(text, str):
        return word_tokenize(text)
    else:
        return [] # Use NLTK's word_tokenize function

def lemmatizator(words):
    lemmatizer = WordNetLemmatizer()

    for pos_tag in ['v', 'n', 'a', 'r']:
        words = [lemmatizer.lemmatize(word, pos=pos_tag) for word in words]

    return words

############################# SENTIMENT ANALYSIS ############################

def compute_emotion_score(lyric, lexicon):
      """
    Compute emotion scores for a given lyric based on a provided lexicon.

    Parameters:
    - lyric (list): A list of words in the lyric.
    - lexicon (dict): A lexicon mapping words to dictionaries of emotion scores.

    Returns:
    - emotion_score (dict): A dictionary containing the summed emotion scores for the given lyric.
      Each emotion is a key, and the corresponding value is the total score for that emotion in the lyric.
    """
      emotion_score = {}
      for word in lyric:
        if word in lexicon:
            for emotion, score in lexicon[word].items():
                if emotion not in emotion_score:
                    emotion_score[emotion] = 0
                emotion_score[emotion] += score
    
    
        return emotion_score
      
vader = SentimentIntensityAnalyzer()      

def vader_wrapper(song_lyrics, mean_sentence=False):
     """
    Calculate the sentiment polarity of song lyrics using the VADER sentiment analysis tool.

    Parameters:
    - song_lyrics (str): The text of the song lyrics to analyze.
    - mean_sentence (bool): If True, calculate the mean sentiment polarity across sentences in the lyrics.
                           If False (default), calculate the overall sentiment polarity of the entire lyrics.

    Returns:
    - float: The sentiment polarity score, ranging from -1.0 (most negative) to 1.0 (most positive).
             Positive scores indicate positive sentiment, negative scores indicate negative sentiment,
             and scores close to 0.0 suggest a more neutral sentiment.
    """
     

     polarity = vader.polarity_scores(song_lyrics)["compound"]
     return polarity

def textblob_wrapper(song_lyrics, mean_sentence=False):
    """
    Calculate the sentiment polarity of song lyrics using TextBlob.

    Parameters:
    - song_lyrics (str): The text of the song lyrics to analyze.
    - mean_sentence (bool): If True, calculate the mean sentiment polarity across sentences in the lyrics.
                           If False (default), calculate the overall sentiment polarity of the entire lyrics.

    Returns:
    - float: The sentiment polarity score, ranging from -1.0 (most negative) to 1.0 (most positive).
             Positive scores indicate positive sentiment, negative scores indicate negative sentiment,
             and scores close to 0.0 suggest a more neutral sentiment.
    """
    polarity = TextBlob(song_lyrics).sentiment.polarity
    return polarity

############################# LOG RATIO ############################
def tag_dict(data):
    tag_dict = {}
    lengths = {}  
    total_len = 0
    for genre in data.tag.unique():
        
        genre_lyrics = data.loc[data.tag == genre].lyrics
        all_lyrics = ' '.join(list(genre_lyrics))
        tokens = word_tokenize(all_lyrics)
        genre_dict = FreqDist(tokens)
        
        # total_words = genre_dict.N()

        # fdist_percentages = FreqDist({word: count / total_words * 100 for word, count in genre_dict.items()})

        
        tag_dict[genre] =  genre_dict 
        lengths[genre] = len(tokens)
        total_len += len(tokens)

    return tag_dict, lengths, total_len

def count_feature(data, column, n = 100):
    
    all_words = ' '.join([word for word in data[column]])   
    tokenized_words = nltk.tokenize.word_tokenize(all_words)   
    fdist = FreqDist(tokenized_words)

    most_common_lyrics = fdist.most_common(n)

    return most_common_lyrics

def check_common_words(tag_dict, freqdist_all):
    # Create a list of FreqDist objects from the dictionary
    freqdists = list(tag_dict.values())

    # Extract the N most common words from the major FreqDist
    common_words = [word for word, count in freqdist_all]

    for word in common_words:
        if not all(word in freqdist for freqdist in freqdists):
            print(f"The word '{word}' from the 20 most common words in the major FreqDist does not appear in all of the other FreqDists.")
    else:
        print('Not a single word from the 20 most common ones appear in any top20 of the tags words.')

def remove_common_words(tag_dict, most_common_keys):
    # Convert most_common_lyrics to a set for efficient lookup
    most_common_set = set(word for word, count in most_common_keys)

    # Create a new dictionary for tags, excluding words in most_common_set
    new_tag_dict = {}
    for genre, fdist in tag_dict.items():
        new_fdist = nltk.FreqDist({word: count for word, count in fdist.items() if word not in most_common_set})
        new_tag_dict[genre] = new_fdist

    return new_tag_dict

def find_common_keys(big_dict):
    # Convert lists of pairs to dictionaries
    dicts = {k: dict(v) for k, v in big_dict.items()}

    # Find all keys
    all_keys = set().union(*(d.keys() for d in dicts.values()))

    # Find which dictionaries contain each key
    key_locations = {key: [dict_name for dict_name, d in dicts.items() if key in d] for key in all_keys}

    # Filter to include only keys that appear in more than one dictionary
    common_keys = {key: locations for key, locations in key_locations.items() if len(locations) > 1}

    return common_keys

def prep_data(data, column, label, n = 100):
    
    # check for
    most_common_keys = count_feature(data, column , n)
    
    print("\nMost Common Keys:\n", most_common_keys)

    # Create dictionaries for all tags
    tags,lengths, total_len= tag_dict(data)
    print("\nDictionary containing fdsits for each tag:\n", tags)

    # # Check if any of the n most common words do not appear in 1 of the tags
    # words_check = check_common_words(tags, most_common_keys)
    # print(f"\nThese are the words that are both importante for a tag and are one of the most frequent in the whole {column} data:\n", words_check)

    # # Remove common words from tags
    # tags = remove_common_words(initial_tags, most_common_keys) 
    # print("\nAfter removing the words that are inside the most common lyrics we get this:\n", tags)

    # Get n most common in each tag
    top_n = {genre: tags[genre].most_common(n) for genre in tags}
    print(f"\nTop {n}:\n", top_n)

    # Print top n for each genre    
    for label_value in data[label].value_counts().keys():
        print("\nGenre:", label_value, f"\nTop {n}:", top_n[label_value])
    
    # return most_common_keys, fdist_percentages, words_check, tags, topn
    
    return most_common_keys, tags, lengths, total_len, top_n

def log_ratio_analysis(top_n, tags, lengths, total_len, distribution, data, label):
    
    # Calculate the total frequency of each word across all data
    total_freq = {}
    for genre, freq_dist in tags.items():
        for word, frequency in freq_dist.items():
            if word not in total_freq:
                total_freq[word] = 0 
            total_freq[word] += frequency

    # Calculate the log ratio for each word in the top N of each genre
    log_ratios = {}
    for genre, freq_dist in top_n.items():
        log_ratios[genre] = {}
        for word, frequency in freq_dist:
            # Subtract the frequency in the current genre from the total frequency
            other_freq = total_freq[word] - frequency
            # Use a small constant to avoid division by zero
            log_ratios[genre][word] = np.log(((frequency + 0.0001) / lengths[genre]) / 
                                             ((other_freq + 0.0001) / (total_len - lengths[genre])))
            
            
    logs = {}
    sorted_logs = {}    

    for genre in data[label].value_counts().keys():
        logs[genre] = sorted(log_ratios[genre].items(), key = lambda x: x[1], reverse=True)
        # Select the top N words for each genre
        sorted_logs[genre] = logs[genre][:distribution[genre]]

    
    return sorted_logs


######################### MODELLING ####################


def score_report(y_train, pred_train , y_val, pred_val):
    print('──────────────────────────────────────────────────────────────────')
    print('                            TRAIN                                                 ')
    print('-----------------------------------------------------------------------------------------------------------')
    print(metrics.classification_report(y_train, pred_train))


    print('─────────────────────────────────────────────────────────────────')
    print('                            VALIDATION                                                 ')
    print('-----------------------------------------------------------------------------------------------------------')
    print(metrics.classification_report(y_val, pred_val))



print('DONE')