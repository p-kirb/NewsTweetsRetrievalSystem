import pandas as pd
import scipy.sparse as sp
import math
import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import normalize


TWEETS_DATAFRAME_PATH = "search_engine/model_components/processed_data.csv"
INPUT_PATH = "search_engine/model_components/"


# VECTOR SPACE ALGORITHM
def vector_space_model(query_tokenized, tweetVectors, idf_dict, term_id, allTweets, N = 10):
    
    """
    Aims to calculate cosine similarities between tf-idf vectors of the documents and query vector. Then chooses top ranked documents.
            Parameters:
                query (array): Array containing tokenized query. 
                tweetVectors (dok_matrix): Sparse matrix containing tf-idf vectors of the documents
                idf_dict (dict): Dictionary of idf values for all terms in corpus.
                allTweets (pandas.DataFrame): Data frame of all tweets.
                N (int): Number of top tweets which are selected in the output.
            Returns:
                results (pandas.series): Data frame of top ranked tweets.      
                
    """

    query_tf = dict(map(lambda token : (token, 1 + math.log(query_tokenized.count(token), 10)), query_tokenized))
    query_tf_idf = np.zeros(len(idf_dict))

    for term in query_tf.keys():
        if term in term_id.keys():
            query_tf_idf[int(term_id[term])] = query_tf[term] * idf_dict[term]
    
    query_tf_idf = query_tf_idf/norm(query_tf_idf)
    tweetVectors = normalize(tweetVectors.tocsr(), norm='l2', axis = 1)

    cosine_similarity = tweetVectors.dot(query_tf_idf)
    doc_id = np.argsort(-cosine_similarity)[0:N]
    results = allTweets["id"][doc_id]

    return results, query_tf_idf
    
# BESTMATCH25 ALGORITHM
def BM25(query_tokenized, factors_matrix, idf_dict, term_id, allTweets, k_3 = 1.5, N = 10):

    """
    Aims to calculate RSV values for each document. Then chooses top ranked documents. The formula depends on the length of the query.
            Parameters:
                query (array): Array containing tokenized query. 
                factors_matrix (dok_matrix): Sparse matrix containing the factors of the term frequency and document length (rows: document id, columns: term id)
                idf_dict (dict): Dictionary of idf values for all terms in corpus.
                allTweets (pandas.DataFrame): Data frame of all tweets.
                k_3 (float): Additional tunning parameter controlling term frequency scaling of the query (used for long queries).
                b (float): Tuning parameter controlling the scaling by document length
            Returns:
                results (pandas.series): Data frame of top ranked tweets.   
                
    """

    query_tf = dict(map(lambda token : (token, query_tokenized.count(token)), query_tokenized))
    query_unique_terms  = query_tf.keys()
    query_idfs = []
    query_tokenized_id = []

    for term in query_unique_terms:
        if term in idf_dict.keys():
            query_idfs.append(idf_dict[term])
            query_tokenized_id.append(term_id[term])

    if len(query_tokenized) <= 6:
        output = np.dot(factors_matrix[:,query_tokenized_id].toarray(), query_idfs)
        doc_id = np.argsort(-output)[0:N]

    else:
        nominator = (k_3+1) * np.array(list(query_tf.values()))
        denominator = k_3 + np.array(list(query_tf.values()))
        query_factor = np.array([x/y for x,y in zip(nominator, denominator)])
        output = np.dot(factors_matrix[:,query_tokenized_id].toarray(),np.multiply(query_idfs, query_factor))
        doc_id = np.argsort(-output)[0:N]

    results = allTweets["id"][doc_id]

    return results

# MIXTURE MODEL
def mixture_model(query_tokenized, parameter_matrix, term_id, allTweets, cf, lam=0.6, N=10):

    """
        Aims to estimate probability of the query occuring in each document documents. Then algorithm selects top ranked documents.
            Parameters:
                tweet_tf (array): Array containing dictionaries of term frequencies of each document. 
                parameter_matrix (dok_matrix): Sparse matrix representing parameters for each term across all documents.
                allTweets (pandas.DataFrame): Data frame of all tweets.
                cf (dict): Dictionary of the terms frequency across the whole corpus.
                lam (float): Lambda parameter [0,1]
                N (int): Number of top tweets which are selected in the output.
            Returns:
                results (poandas.series): Data frame of top ranked tweets.   
                
    """

    query_tf = dict(map(lambda token : (token, query_tokenized.count(token)), query_tokenized))
    query_unique_terms  = query_tf.keys()

    query_tokenized_id = []
    query_cf = []
    for term in query_unique_terms:
        if term in term_id.keys():
            query_cf.append(cf[term])
            query_tokenized_id.append(term_id[term])

    query_tokenized_id = np.array(query_tokenized_id)
    query_cf = np.array(query_cf)

    values = lam * parameter_matrix[:, query_tokenized_id].toarray() + (1-lam) * query_cf/sum(cf.values())
    doc_id = np.argsort(-np.multiply.reduce(values, axis=1))[0:N]
    
    results = allTweets["id"][doc_id]

    return results


def load_data(model_components_path, tweets_dataframe_path):
    # Read necessary data (output of create_components.py)
    data = {}
    data['tweet_vectors'] = sp.load_npz(model_components_path+"documentVectors.npz")

    idf_dict = pd.read_csv(model_components_path+"idfDict.csv", sep=",", keep_default_na=False)
    data['idf_dict'] = dict(zip(idf_dict.term, idf_dict.idf))

    term_id = pd.read_csv(model_components_path+"termIndexDict.csv", sep=",", keep_default_na=False)
    data['term_id'] = dict(zip(term_id.term, term_id["index"]))

    all_tweets = pd.read_csv(tweets_dataframe_path, low_memory=False)
    data['all_tweets'] = all_tweets

    factors_matrix = sp.load_npz(model_components_path+"factorsMatrix.npz")
    data['factors_matrix'] = factors_matrix.todok()

    cf = pd.read_csv(model_components_path+"cfDict.csv", sep=",", keep_default_na=False)
    data['cf'] = dict(zip(cf.term, cf.cf))
    
    parameter_matrix = sp.load_npz(model_components_path+"parameterMatrix.npz")
    data['parameter_matrix'] = parameter_matrix.todok()

    del idf_dict, term_id, all_tweets, factors_matrix, cf, parameter_matrix

    return data