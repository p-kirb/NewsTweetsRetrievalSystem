import pandas as pd
import scipy.sparse as sp
import math
import time
from ast import literal_eval
import numpy as np


rowIndex=0


def loadTokenized(data_path):
    print("Reading preprocessed CSV...")
    start = time.time()
    returnDF = pd.read_csv(data_path)
    #  needed to convert string of list to plain list
    returnDF["preprocessed_text"] = list(map(lambda tweet : literal_eval(tweet),returnDF["preprocessed_text"]))
    print(f"Took {time.time()-start} seconds.\n")
    return returnDF


def calcLogWeightedTF(tokenizedTweets):
    """
    calculate the Log Weighted Term Frequencies of terms in a series of processed tweets.

            Parameters:
                tokenizedTweets (pandas.Series): Series of arrays - each entry is an array representing a tokenized tweet.
            Returns:
                returnSeries (pandas.Series): Series of dictionaries - each series entry is a dictionary containing, for that row, the unique term:log-weighted TF score pairs.
    """
    #getting log weighted tf scores for each term
    print("Calculating weighted tf scores...")
    start = time.time()
    def logWeightedTF(tweet):
        return dict(
                    map(
                        lambda token : (token, 1 + math.log(tweet.count(token), 10)), tweet
                        )
                    )
    # returnSeries = list(map(lambda tweet : dict(map(lambda token : (token, 1 + math.log(tweet.count(token), 10)), tweet)), tokenizedTweets))
    returnSeries = list(map(logWeightedTF, tokenizedTweets))
    print(f"Took {time.time()-start} seconds.\n")
    return returnSeries


def calcUnweightedTF(tokenizedTweets):
    """
    calculate the Unweighted Term Frequencies of terms in a series of processed tweets.

            Parameters:
                tokenizedTweets (pandas.Series): Series of arrays - each entry is an array representing a tokenized tweet.
            Returns:
                returnSeries (pandas.Series): Series of dictionaries - each series entry is a dictionary containing, for that row, the unique term:TF score pairs.
    """
    print("Calculating tf scores...")
    start = time.time()
    returnSeries = list(
        map(
            lambda tweet : dict(
                map(
                    lambda token : (token, tweet.count(token)), tweet
                    )
                ), tokenizedTweets
            )
        )
    print(f"Took {time.time()-start} seconds.\n")
    return returnSeries


def calcDFScores(termFreqSeries):
    """
    calculate the Document Frequencies of terms from a series of term:TF dictionaries.

            Parameters:
                tokenizedTweets (pandas.Series): series of term:TF dictionaries.
            Returns:
                dfDict (dict): dictionary holding term:document frequency pairs for all terms in the corpus.
    """
    dfDict = {}

    def getDF(rowDict):
        for term in rowDict:
            if not term in dfDict:
                dfDict[term] = 1
            else:
                dfDict[term] += 1

    print("Accumulating df scores...")
    start = time.time()
    termFreqSeries.agg([getDF])
    print(f"Took {time.time()-start} seconds.\n")
    return dfDict


def calcIDFScores(dfDict, noOfDocs):
    """
    Calculates idf scores for each term, while also making a term order dictionary
    Term order dictionary stores term:(index in list) pairs to allow for reverese lookup of the index of a term in the model vector for faster production of vectors.
    (this means the program doesnt have to do a linear search every time it needs to know a term's position when creating the tf-idf matrix)
            Parameters:
                dfDict (dict): Dictionary containing term:df pairs for all terms in the corpus.
                noOfDocs (int): Total number of documents in the corpus
            Returns:
                idfDict (dict): Dictionary containing term:idf pairs for all terms in the corpus.
                termIndexDict (dict): Dictionary containing term:index pairs for fast reverse lookup.
    """
    print("Calculating idf scores...")
    start = time.time()
    idfDict = {}
    termIndexDict = {}
    termIndex = 0
    for term, df in dfDict.items():
        idfDict[term] = math.log(noOfDocs/df, 10)
        termIndexDict[term] = termIndex
        termIndex += 1

    print(f"Took {time.time()-start} seconds.\n")
    return idfDict, termIndexDict


def generateTweetVectors(tweetTFDicts, idfDict, termIndexDict):
    """
    Generates a tf-idf matrix that represents the corpus.
    Each row represents a tweet, each column represents a unique term from the entire corpus.
    Each entry is the tf-idf score of the corresponding word in the context of the corresponding tweet.
            Parameters:
                tweetTFDicts (pandas.Series): Series of dictionaries - each series entry is a dictionary containing, for that row, the unique term:log-weighted TF score pairs.
                idfDict (dict): Dictionary containing term:idf pairs for all terms in the corpus.
                termIndexDict (dict): Dictionary containing term:index pairs for fast reverse lookup.
            Returns:
                tweetVectors (scipy.sparse.dok_matrix): Sparse tf-idf matrix representation of the corpus.
    """
    termCount = len(idfDict)
    tweetCount = tweetTFDicts.size

    #generating document vectors
    print(f"Generating tweet vectors ({tweetCount} x {termCount} matrix)")
    start = time.time()
    tweetVectors = sp.dok_matrix((tweetCount, termCount), dtype="float64")

    def populateVector(tweetTerms):
        global rowIndex
        for term, tfWeight in tweetTerms.items():
            tweetVectors[rowIndex, termIndexDict[term]] = tfWeight * idfDict[term]
        rowIndex += 1

    tweetTFDicts.agg([populateVector])          #agg marginally quicker than pandarallel (~30s vs ~33s)

    print(f"Took {time.time()-start} seconds.\n")

    return tweetVectors


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
    L_d = list(map(lambda x: sum(x.values()),tweet_tf))
    L_avg = np.sum(L_d)/tweet_count

    factors_matrix = sp.dok_matrix((tweet_count, term_count), dtype="float64")
    
    for doc_id in range(tweet_count):
        for term, tf in tweet_tf[doc_id].items():
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
        d = len(tweet_tf[doc_id].keys())
        for term, tf in tweet_tf[doc_id].items():
            parameter_matrix[doc_id, term_id[term]] = tf/d
    return parameter_matrix