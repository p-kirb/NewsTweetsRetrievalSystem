from helpers import remove_duplicate_tweets


DATA_PATH = 'data/processed_data/processed_data.csv'
DUPLICATE_COL = 'tweet'


remove_duplicate_tweets(DATA_PATH, DUPLICATE_COL, overwrite=True)