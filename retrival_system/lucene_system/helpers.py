import pandas as pd
import os
# from spellchecker import SpellChecker
# from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# from nltk import word_tokenize
# from time import time
import re


def preprocess_string(text, stopwords=stopwords.words('english'), lemmatizer=WordNetLemmatizer()):
    '''
    Preprocess the input string, applying the following steps:
        1. Lowercase the text
        2. Remove URLs
        3. Remove punctuation
        4. Tokenize the text
        5. Remove stopwords
        6. Spell check the text
        7. Lemmatize the text

            Parameters:
                text (str): The text to be preprocessed.
                stopwords (list): List of stopwords to be removed.
                lemmatizer (nltk.stem.WordNetLemmatizer): Lemmatizer to be used.

            Returns:
                result (list): Preprocessed text.
    '''
    text = text.lower()
    text = remove_whitespace(text)
    text = remove_urls(text)
    text = remove_punctuation(text)
    text = tokenize_text(text)
    text = remove_stopwords(text, stopwords)
    text = spell_check(text)
    text = [lemmatizer.lemmatize(word) for word in text]
    return text


def extract_urls(text):
    '''
    Extract URLs from the text (specifically, from the tweet text - https://t.co/...).
    
            Parameters:
                text (str): The text to be processed.
                
            Returns:
                urls (list): List of URLs extracted from the text.
    '''
    url_pattern = re.compile(r'https?://\S+|t\.\S+')
    url = re.search(url_pattern, text)
    if url:
        url = url.group()
    return url


def join_data(data_dir, output_dir=None):
    """
    Join data from multiple .csv files into a single file.

            Parameters:
                data_dir (str): Path to the directory containing the data files.
                output_dir (str): Path to the directory where the output file will be saved.
                    default: None (the output file will not be saved)
                
            Returns:
                joined_data (pandas.DataFrame): Data from all files joined into a single DataFrame.
    """

    files_to_join = os.listdir(data_dir)  # Get the list of files in the data directory
    print('Files to join: {}'.format(files_to_join))
    joined_data = pd.DataFrame()  # Create an empty DataFrame to store the joined data
    # Load and join the files
    start_time = time()
    for filename in files_to_join:
        print('\t Loading file: {}'.format(filename))
        file_path = os.path.join(data_dir, filename)
        data = pd.read_csv(file_path, low_memory=False)
        joined_data = pd.concat([joined_data, data])
    joined_data.reset_index(drop=True, inplace=True)
    end_time = time()
    print(f'Data successfully joined in {end_time - start_time} seconds!')
    # Save the joined data
    if output_dir:
        output_path = os.path.join(output_dir, 'joined_data.csv')
        joined_data.to_csv(output_path, index=False)
        print('\t Joined data saved to: {}'.format(output_path))

    return joined_data


def remove_whitespace(text):
    '''
    Remove whitespace from the text.

            Parameters:
                text (str): The text to be processed.

            Returns:
                text (str): The processed text.
    '''
    return " ".join(text.split())


def remove_urls(text):
    '''
    Remove URLs from the text. Specifically, remove URLs from the tweet text - https://t.co/...

            Parameters:
                text (str): The text to be processed.

            Returns:
                text (str): The processed text without twitter URLs.
    '''
    url_pattern = re.compile(r'https?://\S+|t\.\S+')
    return re.sub(url_pattern, '', text)


def remove_punctuation(text):
    '''
    Remove punctuation from the text, using NLTK's RegexpTokenizer.

            Parameters:
                text (str): The text to be processed.

            Returns:
                text (str): The processed text.
    '''
    tokenizer = RegexpTokenizer(r"\w+")
    lst = tokenizer.tokenize(text)
    return " ".join(lst)


def tokenize_text(text):
    '''
    Tokenize the text using NLTK's word_tokenize.

            Parameters:
                text (str): The text to be tokenized.

            Returns:
                text (list): The tokenized text.
    '''
    return word_tokenize(text)


def remove_stopwords(text, stopwords):
    '''
    Remove stopwords from the text.
    
            Parameters:
                text (list): Tokenized text.
                stopwords (list): List of stopwords to be removed.
                
            Returns:
                text (list): The text without stopwords.
    '''
    result = []
    for token in text:
        if token not in stopwords:
            result.append(token)
            
    return result


def spell_check(text):
    '''
    Spell check the text using the SpellChecker library.

            Parameters:
                text (list): List of tokens to be spell checked.

            Returns:
                text (list): List of tokens after spell checking and correction.
    '''
    result = []
    spell = SpellChecker(case_sensitive=True)
    for word in text:
        correct_word = spell.correction(word)
        result.append(correct_word if correct_word is not None else word)
    
    return result


def lemmatize_text(text, lemmatizer=WordNetLemmatizer()):
    '''
    Lemmatize the text using NLTK's WordNetLemmatizer.
    
            Parameters:
                text (list): List of tokens to be lemmatized.
                lemmatizer (nltk.stem.WordNetLemmatizer): Lemmatizer to be used.
                
                Returns:
                    text (list): List of tokens after lemmatization.
    '''
    return [lemmatizer.lemmatize(word) for word in text]


def remove_duplicate_tweets(data_path, dupliacte_col, overwrite=False):
    """
    Remove duplicate tweets from the data.

            Parameters:
                data_path (str): Path to the data file.
                dupliacte_col (str): Name of the column containing the tweet text.
                
            Returns:
                data (pandas.DataFrame): Data with duplicate tweets removed.
    """

    data = pd.read_csv(data_path, low_memory=False)
    print('Data shape before removing duplicates: {}'.format(data.shape))
    # remove urls from tweet text so that duplicates are based on the tweet text only
    data['duplicate'] = data[dupliacte_col].apply(lambda x: remove_urls(x))
    data.drop_duplicates(subset=['duplicate'], inplace=True)
    data.drop(columns=['duplicate'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    print('Data shape after removing duplicates: {}'.format(data.shape))

    if overwrite:
        data.to_csv(data_path, index=False)
        print('Data saved to: {}'.format(data_path))

    return data