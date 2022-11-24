from helpers import join_data
from helpers import preprocess_string, extract_urls
import pandas as pd
from nltk.corpus import stopwords
from pandarallel import pandarallel
import time

OUTPUT_PATH = 'data/processed_data/processed_data.csv'

# initialiaze pandarallel
pandarallel.initialize(progress_bar=True)

input_data = join_data('data/input_data', 'data/joined_data')
# loaded_data = pd.read_csv('joined_data/joined_data.csv', low_memory=False)

# Preprocess the data
def preprocess_data(data, text_column, cols_to_get, stopwords=stopwords.words('english')):
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
    # Preprocess the text
    data = data[cols_to_get]
    data['preprocessed_text'] = data[text_column].parallel_apply(preprocess_string)
    data['tweet_url'] = data[text_column].parallel_apply(extract_urls)

    print('Preprocessing completed in {} seconds!'.format(time.time() - start_time))

    return data

processed_data = preprocess_data(input_data, 'tweet', cols_to_get=['id', 'tweet', 'urls'])

for i in range(20):
    print('Tweet: \n \t', processed_data['tweet'][i])
    print('URL: \n \t', processed_data['tweet_url'][i])
    print('Processed tweet: \n \t', processed_data['preprocessed_text'][i])

data_to_save = processed_data

# Save the preprocessed data
processed_data.to_csv(OUTPUT_PATH, index=False)