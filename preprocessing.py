from helpers import join_data
from helpers import spell_check, remove_stopwords, remove_punctuation, remove_urls
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pandarallel import pandarallel
import time

OUTPUT_PATH = 'data/processed_data/processed_wo_spelling_data.csv'

# initialiaze pandarallel
pandarallel.initialize()

data = join_data('data/input_data', 'data/joined_data')
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
    print(data.head())
    start_time = time.time()
    # Select columns
    data = data[cols_to_get]
    # Unique values in rows
    print('Removing duplicates...')
    print('Number of rows before removing duplicates: {}'.format(data.shape[0]))
    data.drop_duplicates(inplace=True)
    data.reset_index(drop=True, inplace=True)
    print('Number of rows after removing duplicates: {}'.format(data.shape[0]))
    # Remove URLs
    print('Removing URLs...')
    data['processed'] = data[text_column].parallel_apply(remove_urls)
    print(data['processed'].head())   
    end_time = time.time()
    # Convert to lowercase
    print('Converting to lowercase...')
    data['processed'] = data['processed'].str.lower()
    print(data['processed'].head())
    # Remove punctuation
    print('Removing whitespaces...')
    data['processed'] = data['processed'].parallel_apply(lambda x: ' '.join(x.split()))
    print(data['processed'].head())
    # Tokenize the text
    print('Tokenizing...')
    data['processed'] = data['processed'].parallel_apply(lambda x: word_tokenize(x))
    print(data['processed'].head())
    # Spelling Correction
    # print('Correcting spelling...')
    # data['processed'] = data['processed'].parallel_apply(spell_check)
    # print(data['processed'].head())
    # Remove stopwords
    print('Removing stopwords...')
    data['processed'] = data['processed'].parallel_apply(remove_stopwords, stopwords=stopwords)
    print(data['processed'].head())
    # Remove punctuation
    print('Removing punctuation...')
    data['processed'] = data['processed'].parallel_apply(remove_punctuation)
    print(data['processed'].head())
    # Lemmatize the text (prefered to stemming, as it does morphological analysis of the words)
    print('Lemmatizing...')
    data['processed'] = data['processed'].parallel_apply(lambda x: [WordNetLemmatizer().lemmatize(word=word) for word in x])
    print(data['processed'].head())


    print(f'Data successfully preprocessed in {end_time - start_time} seconds!')

    return data

processed_data = preprocess_data(data, 'tweet', cols_to_get=['id', 'tweet', 'urls'])

for i in range(20):
    print('Tweet: \n \t', processed_data['tweet'][i])
    print('Processed tweet: \n \t', processed_data['processed'][i])

data_to_save = processed_data

# Save the preprocessed data
processed_data.to_csv(OUTPUT_PATH, index=False)