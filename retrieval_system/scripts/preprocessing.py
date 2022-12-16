from functions.preprocessing.helpers import preprocess_string
from functions.preprocessing.helpers import join_data
from functions.preprocessing.helpers import remove_duplicate_tweets
from functions.preprocessing.helpers import extract_urls
from pandarallel import pandarallel
import time
import os


DATA_INPUT_FOLDER = 'data/01_input_data/'
DATA_JOINED_OUTPUT_FOLDER = 'data/02_joined_data/'
DATA_PROCESSED_OUTPUT_FOLDER = 'data/03_processed_data/'

DATA_TEXT_COLUMN = 'tweet'
PROCESSED_DATA_COLUMNS = ['id', 'tweet', 'urls']
CHECK_SPELLING = True


def preprocess_data(data, text_column, cols_to_get, check_spelling=True):
    """
    Preprocess the data by tokenizing the text and removing the punctuation.

            Parameters:
                data (pandas.DataFrame): Data to be preprocessed.
                text_column (str): Name of the column containing the text to be processed.
                stopwords (list): List of stopwords to be removed from the text.

            Returns:
                preprocessed_data (pandas.DataFrame): Preprocessed data.
    """
    print('Starting preprocessing...')
    start_time = time.time()

    # Preprocess the text column
    data = data[cols_to_get]

    print(f'\tPreprocessing {text_column} column...')
    data['preprocessed_text'] = data[text_column].parallel_apply(preprocess_string, check_spelling=check_spelling)

    print(f'\tExtracting URLs from {text_column} column...')
    data['tweet_url'] = data[text_column].parallel_apply(extract_urls)

    print('\tRemoving duplicates...')
    data = remove_duplicate_tweets(data=data, dupliacte_col=text_column)

    print('Preprocessing completed in {} seconds!'.format(time.time() - start_time))

    return data


if __name__ == '__main__':
    # Initialize pandarallel -- multiprocessing
    pandarallel.initialize(progress_bar=True)
    
    # Load and join the data
    input_data = join_data(DATA_INPUT_FOLDER, DATA_JOINED_OUTPUT_FOLDER)

    # Preprocess the data
    processed_data = preprocess_data(
        data=input_data, 
        text_column=DATA_TEXT_COLUMN, 
        cols_to_get=PROCESSED_DATA_COLUMNS,
        check_spelling=CHECK_SPELLING
        )

    # Save the preprocessed data
    processed_data.to_csv(DATA_PROCESSED_OUTPUT_FOLDER+'processed_data.csv', index=False)