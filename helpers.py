import pandas as pd
import os
from spellchecker import SpellChecker
from nltk.tokenize import RegexpTokenizer
from time import time
import re

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


def spell_check(text):
    
    result = []
    spell = SpellChecker(case_sensitive=True)
    for word in text:
        correct_word = spell.correction(word)
        result.append(correct_word if correct_word is not None else word)
    
    return result


def remove_stopwords(text, stopwords):
    result = []
    for token in text:
        if token not in stopwords:
            result.append(token)
            
    return result


def remove_punctuation(text):
    
    tokenizer = RegexpTokenizer(r"\w+")
    lst=tokenizer.tokenize(' '.join(text))
    return lst


def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|t\.\S+')
    return url_pattern.sub(r'', text)