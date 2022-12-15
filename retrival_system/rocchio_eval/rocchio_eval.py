from helpers import preprocess_string
from rocchio import rocchio_algorithm
from search_models import load_data, vector_space_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import json


TWEETS_DATAFRAME_PATH = "../data/processed_data/processed_data.csv"
MODEL_COMPOMENTS_PATH = "../data/model_components/"

relevant_queries = pd.read_csv("relevant_queries.csv", sep=";")
print(relevant_queries.head())
loaded_data = load_data(MODEL_COMPOMENTS_PATH, TWEETS_DATAFRAME_PATH)
tweets_df = loaded_data['all_tweets']
tf_idf = loaded_data['tweet_vectors'].tocsr()
lucene_result = {
    'Thailand cave rescue': 25, 
    'Notre-dame fire': 22, 
    'US capitol building storming': 23, 
    'water found on Mars': 16, 
    'El Chapo escapes prison': 17
    }

# total_df = pd.DataFrame()
total_dict = {}
for i, (query, n) in enumerate(lucene_result.items()):
    processed_query = preprocess_string(query)

    results, old_tf_idf = vector_space_model(
                    processed_query, 
                    loaded_data['tweet_vectors'], 
                    loaded_data['idf_dict'], 
                    loaded_data['term_id'], 
                    loaded_data['all_tweets'], 
                    N=n
                )
    relevant_list = np.array([True if tweets_df[tweets_df['id'] == result]['tweet'].iloc[0] in list(relevant_queries[query]) else False for result in results])
    relevant_ids = results[relevant_list].index
    nonrelevant_ids = results[~relevant_list].index

    rel_docs = tf_idf[relevant_ids, :]
    nonrel_docs = tf_idf[nonrelevant_ids, :]
    new_query = rocchio_algorithm(old_tf_idf, rel_docs, nonrel_docs)

    tweetVectors = normalize(tf_idf, norm='l2', axis = 1)
    cosine_similarity = tweetVectors.dot(new_query)
    doc_id = np.argsort(-cosine_similarity)[0:n]
    new_results = loaded_data['all_tweets']["id"][doc_id]

    total_dict[query] = list(new_results)


jay = json.dumps(total_dict)
with open('rocchio_results.json', 'w') as f:
    f.write(jay)

with open('rocchio_results.json', 'r') as f:
    x = json.load(f)