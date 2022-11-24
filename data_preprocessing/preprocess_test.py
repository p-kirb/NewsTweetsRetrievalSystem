from helpers import preprocess_string
from nltk.corpus import stopwords
import pytest
import pandas as pd
from ast import literal_eval

DATA_PATH = 'data/processed_data/processed_data.csv'
STRING_COL = 'tweet'
PREPROCESSED_COL = 'preprocessed_text'

# Load the data
data = pd.read_csv('data/processed_data/processed_data.csv', low_memory=False)

# Sample data for testing
data_for_test = data.sample(10)

class TestPreprocessingString():
    def test_one(self):
        test_num = 1
        test_string = data_for_test[STRING_COL].iloc[test_num - 1]
        test_func_result = preprocess_string(test_string)
        dataframe_result = data_for_test[PREPROCESSED_COL].iloc[test_num - 1]
        assert test_func_result == literal_eval(dataframe_result)

    def test_two(self):
        test_num = 2
        test_string = data_for_test[STRING_COL].iloc[test_num - 1]
        test_func_result = preprocess_string(test_string)
        dataframe_result = data_for_test[PREPROCESSED_COL].iloc[test_num - 1]
        assert test_func_result == literal_eval(dataframe_result)

    def test_three(self):
        test_num = 3
        test_string = data_for_test[STRING_COL].iloc[test_num - 1]
        test_func_result = preprocess_string(test_string)
        dataframe_result = data_for_test[PREPROCESSED_COL].iloc[test_num - 1]
        assert test_func_result == literal_eval(dataframe_result)

    def test_four(self):
        test_num = 4
        test_string = data_for_test[STRING_COL].iloc[test_num - 1]
        test_func_result = preprocess_string(test_string)
        dataframe_result = data_for_test[PREPROCESSED_COL].iloc[test_num - 1]
        assert test_func_result == literal_eval(dataframe_result)

    def test_five(self):
        test_num = 5
        test_string = data_for_test[STRING_COL].iloc[test_num - 1]
        test_func_result = preprocess_string(test_string)
        dataframe_result = data_for_test[PREPROCESSED_COL].iloc[test_num - 1]
        assert test_func_result == literal_eval(dataframe_result)