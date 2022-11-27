import pandas as pd
import scipy.sparse as sp
import math
import time
from csv import DictReader
import numpy as np
from numpy.linalg import norm
import ast
from sklearn.preprocessing import normalize
from data_preprocessing.helpers import preprocess_string
from build_index import calcUnweightedTF



PATH = "data/"

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
    results = allTweets["tweet"][doc_id]

    return results

# BESTMATCH25 ALGORITHM
def BM25(query_terms, factors_matrix, idf_dict, term_id, allTweets, k_3 = 1.5, N = 10):

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

    query_tf = dict(map(lambda token : (token, query_terms.count(token)), query_terms))
    query_unique_terms  = query_tf.keys()
    query_idfs = []
    query_terms_id = []

    for term in query_unique_terms:
        if term in idf_dict.keys():
            query_idfs.append(idf_dict[term])
            query_terms_id.append(term_id[term])

    if len(query_terms) <= 6:
        output = np.dot(factors_matrix[:,query_terms_id].toarray(), query_idfs)
        doc_id = np.argsort(-output)[0:N]

    else:
        nominator = (k_3+1) * np.array(list(query_tf.values()))
        denominator = k_3 + np.array(list(query_tf.values()))
        query_factor = np.array([x/y for x,y in zip(nominator, denominator)])
        output = np.dot(factors_matrix[:,query_terms_id].toarray(),np.multiply(query_idfs, query_factor))
        doc_id = np.argsort(-output)[0:N]

    results = allTweets["tweet"][doc_id]

    return results

# MIXTURE MODEL
def mixture_model(query_terms, parameter_matrix, term_id, allTweets, cf, lam=0.5, N=10):

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

    query_tf = dict(map(lambda token : (token, query_terms.count(token)), query_terms))
    query_unique_terms  = query_tf.keys()

    query_terms_id = []
    query_cf = []
    for term in query_unique_terms:
        if term in term_id.keys():
            query_cf.append(cf[term])
            query_terms_id.append(term_id[term])

    query_terms_id = np.array(query_terms_id)
    query_cf = np.array(query_cf)

    values = lam * parameter_matrix[:, query_terms_id].toarray() + (1-lam) * query_cf/sum(cf.values())
    doc_id = np.argsort(-np.multiply.reduce(values, axis=1))[0:N]
    
    results = allTweets["tweet"][doc_id]

    return results

# ADDITIONAL COMPONENTS REQUIRED TO RUN ALGORITHMS

# FACTORS OF THE RSV EQUATION (BM25)
def factoring(tweet_tf, idf_dict, term_id, k=1.5, b=0.75):

    """ 
    Aims to calculate factors of the term frequency and document length which are used by the BM25 algorithm. 
    It is constant across document. Calculate once so algorithms could extract relevant data.
            Parameters:
                tweet_tf (array): Array containing dictionaries of term frequencies of each document. 
                idf_dict (dict): Dictionary of idf values for all terms in corpus.
                k (float): Tunning parameter controlling the document term frequency scaling.
                N (int): Number of top tweets which are selected in the output.
            Returns:
                results (dok_matrix): Sparse matrix containing the factors of the term frequency and document length (rows: document id, columns: term id)
                
    """
    tweet_count = len(tweet_tf)
    term_count = len(idf_dict)
    L_d = list(map(lambda x: sum(ast.literal_eval(x).values()),tweet_tf))
    L_avg = np.sum(L_d)/tweet_count

    factors_matrix = sp.dok_matrix((tweet_count, term_count), dtype="float64")
    
    for doc_id in range(tweet_count):
        for term, tf in ast.literal_eval(tweet_tf[doc_id]).items():
            factors_matrix[doc_id, term_id[term]] = ((k+1) * tf)/(k * ((1-b) + b * (L_d[doc_id]/L_avg)) + tf)
    return factors_matrix

# TERM FREQUENCY FOR THE WHOLE COLLECTION (MIXTURE MODEL)
def CFt(unweighted_tf):

    """
    Aims to calculate terms frequency across the whole corpus.
            Parameters:
                unweighted_tf (array): Array containing term frequency dictionaries of each document.
            Returns:
                results (dict): Dictionary of the terms frequency across the whole corpus.        
    """

    cf = {}
    for token_set in unweighted_tf:
        token_set = ast.literal_eval(token_set)
        for token in token_set.keys():
            if token not in cf.keys():
                cf[token] = token_set[token]
            else:
                cf[token] += token_set[token]
    return cf

# MAXIMUM LIKELIHOOD ESTIMATES (MIXTURE MODEL)
def parameter_estimation(tweet_tf, idf_dict, term_id):

    """
    Aims to calculate parameter estimates for each term across all documents so that mixture_model algorithm will extract columns related to the query.
            Parameters:
                tweet_tf (array): Array containing dictionaries of term frequencies of each document. 
                idf_dict (dict): Dictionary of idf values for all terms in corpus.
            Returns:
                parameter_matrix (dok_matrix)): Sparse matrix representing parameters for each term across all documents.
                
    """

    tweet_count = len(tweet_tf)
    term_count = len(idf_dict)
    parameter_matrix = sp.dok_matrix((tweet_count, term_count), dtype="float64")

    for doc_id in range(tweet_count):
        d = len(ast.literal_eval(tweet_tf[doc_id]).keys())
        for term, tf in ast.literal_eval(tweet_tf[doc_id]).items():
            parameter_matrix[doc_id, term_id[term]] = tf/d
    return parameter_matrix

def runAll(query, run_components=False, n=10):

    # READ REQUIRED DATA (FROM BUILD INDEX)
    tweetVectors = sp.load_npz(PATH+"documentVectors.npz")
    idf_dict = pd.read_csv(PATH+"idfDict.csv", sep=",", keep_default_na=False)
    idf_dict = dict(zip(idf_dict.term, idf_dict.idf))
    term_id = pd.read_csv(PATH+"termIndexDict.csv", sep=",", keep_default_na=False)
    term_id = dict(zip(term_id.term, term_id["index"]))
    allTweets = pd.read_csv(PATH+"processed_data/processed_data.csv", low_memory=False)

    # Calculate unweighted scores
    allTweets["unweightedTf"] = calcUnweightedTF(allTweets["preprocessed_text"])

    if run_components == True:
        # Calculate factor matrix and save as sparse matrix
        factors_matrix = factoring(allTweets["unweightedTf"], idf_dict, term_id, k=1.5, b=0.75)
        sp.save_npz(PATH+"factorsMatrix.npz", factors_matrix.tocoo())
        
        # Calculate parameter matrix and save as sparse matrix
        parameter_matrix = parameter_estimation(allTweets["unweightedTf"], idf_dict, term_id)
        sp.save_npz(PATH+"parameterMatrix.npz", parameter_matrix.tocoo())

        # Calculate term frequency dictionary across whole corpus
        cf = CFt(allTweets["unweightedTf"])
        cfDataFrame = pd.DataFrame(cf.items(), columns=["term", "tf"])
        cfDataFrame.to_csv(PATH+"cfDict.csv", encoding="utf-8", index=False)

    else:
        # Load components
        factors_matrix = sp.load_npz(PATH+"factorsMatrix.npz")
        factors_matrix = factors_matrix.todok()

        cf = pd.read_csv(PATH+"cfDict.csv", sep=",", keep_default_na=False)
        cf = dict(zip(cf.term, cf.tf))

        parameter_matrix = sp.load_npz("data/parameterMatrix.npz")
        parameter_matrix = parameter_matrix.todok()

    # Query tokenization
    query_tokenized = preprocess_string(query)
    
    # Modelling
    vsm = vector_space_model(query_tokenized, tweetVectors, idf_dict, term_id, allTweets, N=n)
    print("Vector space model results:")
    print(vsm)
    
    bm25 = BM25(query_tokenized, factors_matrix, idf_dict, term_id, allTweets, k_3=1.5, N=n)
    print("BestMatch25 model results:")
    print(bm25)

    mm = mixture_model(query_tokenized, parameter_matrix, term_id, allTweets, cf, lam=0.5, N=n)
    print("Mixture model results:")
    print(mm)

    return vsm, bm25, mm

query = "Poland attacked"


vsm, bm25, mm = runAll(query, run_components=False, n=10)

