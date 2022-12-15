import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from search_engine.search_models import load_data, vector_space_model, BM25, mixture_model
from search_engine.helpers import preprocess_string 
from lucene_system.lucene_model import lucene_query
import pandas as pd

TWEETS_DATAFRAME_PATH = "search_engine/processed_data.csv"
MODEL_COMPOMENTS_PATH = "search_engine/model_components/"

query = "Thailand cave rescue"

# Function which runs all models
def running_results(query, loaded_data, n):

    vsm_results = vector_space_model(
                preprocess_string(query), 
                loaded_data['tweet_vectors'], 
                loaded_data['idf_dict'], 
                loaded_data['term_id'], 
                loaded_data['all_tweets'], 
                N=n)
    
    bm25_results = BM25(
                preprocess_string(query), 
                loaded_data['factors_matrix'], 
                loaded_data['idf_dict'], 
                loaded_data['term_id'], 
                loaded_data['all_tweets'], 
                k_3=1.5, 
                N=n) 

    mm_results = mixture_model(
                preprocess_string(query), 
                loaded_data['parameter_matrix'], 
                loaded_data['term_id'], 
                loaded_data['all_tweets'], 
                loaded_data['cf'], 
                lam=0.5, 
                N=n)
    
    # lucene_results = lucene_query(
    #                 preprocess_string(queries[query_id]), 
    #                 n, 
    #                 printResults=False)

    results_dict = {"vsm": vsm_results, "bm25": bm25_results, "mm": mm_results, "lucene": mm_results}#lucene_results["id"]}

    return results_dict

# Function which determines if the output is relevant (lucene treated as a benchamrk)
def benchmarking(results):

    n = len(results["vsm"])
    listed_lucene = list(results["lucene"])
    query_dict = {"vsm": 0, "bm25": 0, "mm": 0}

    for model in query_dict.keys():
        relevance_vector = np.zeros(n)
        
        for result_id in range(n):
            if results[model].iloc[result_id] in listed_lucene:
                relevance_vector[result_id] = 1

        query_dict[model] = relevance_vector

    return query_dict

# # Function which calculates average precision per query and model
def average_precision(relevance_dict, lucene_n):
    avg_p_dict = {}
    for model in relevance_dict.keys():
        relevant_array = relevance_dict[model]
        matrix = np.zeros([len(relevant_array),2])
        nominator = 0
        denominator = 0

        for index in range(len(relevant_array)):
            denominator += 1
            if relevant_array[index] == 1:
                nominator += 1
                matrix[index][0] = nominator/lucene_n # Recall
                matrix[index][1] = nominator/denominator # Precision
    
        avg_p = sum(matrix[:,1])/lucene_n
        avg_p_dict[model] = avg_p
    return avg_p_dict

# Function which gives a matrix with recall and precision values
def precision_recall_matrix(relevance_array, lucene_n):
    
    matrix = np.zeros([len(relevance_array),2])
    nominator = 0
    denominator = 0

    for index in range(len(relevance_array)):
        denominator += 1
        if relevance_array[index] == 1:
            nominator += 1
            matrix[index][0] = nominator/lucene_n # Recall
            matrix[index][1] = nominator/denominator # Precision
        else:
            matrix[index][0] = nominator/lucene_n 
            matrix[index][1] = nominator/denominator

    return matrix

# Provides PR curves for each query comparing models
def pr_curve(relevance_dict, lucene_n):

    n = len(relevance_dict.keys())
    pr_matrix = {model: 0 
                    for model in relevance_dict.keys()}

    for model in relevance_dict.keys():
        pr_matrix[model] = precision_recall_matrix(relevance_dict[model], lucene_n)

    pr_plot = plt.plot(pr_matrix["vsm"][:,0], pr_matrix["vsm"][:,1], '-o', 
                pr_matrix["bm25"][:,0], pr_matrix["bm25"][:,1], '-o', 
                pr_matrix["mm"][:,0], pr_matrix["mm"][:,1], '-o')

    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.legend(['Vector Space Model', 'BestMatch25 Model', "Mixture Model"])

    return pr_plot


def run_all(query, MODEL_COMPOMENTS_PATH, TWEETS_DATAFRAME_PATH, n):

    loaded_data = load_data(MODEL_COMPOMENTS_PATH, TWEETS_DATAFRAME_PATH)
    results_dict = running_results(query, loaded_data, n)
    query_dict = benchmarking(results_dict)
    avg_p = average_precision(query_dict, n)
    pr_plot = pr_curve(query_dict, n)
    
    return avg_p, pr_plot
