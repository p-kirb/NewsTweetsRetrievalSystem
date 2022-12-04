import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from run_query import *

INPUT_PATH = "data/model_components/"

q1 = "Crocodiles breeding"
q2 = "Michael Schumacher paralized"
q3 = "John Paul II saint"
queries = [q1,q2,q3]

data = load_data(INPUT_PATH)

def running_result(queries, data):
    results_dict = {}
    for query_id in range(len(queries)):
        vsm_results, bm25_results, mm_results = run_all_models(q1, data, n=10)
        results_dict[query_id] = {"vsm": vsm_results, "bm25": bm25_results, "mm": mm_results, "lucene": mm_results}
    return results_dict

results_dict = running_result(queries, data)

def benchmarking(results):
    n = len(results[0]["vsm"])
    relevance_dict = {}
    for query_id in range(len(results.keys())):
        listed_lucene = list(results[query_id]["lucene"])
        query_dict = {"vsm": 0, "bm25": 0, "mm": 0}
        for model in query_dict.keys():
            relevance_vector = np.zeros(n)
            for result_id in range(n):
                if results[query_id][model].iloc[result_id] in listed_lucene:
                    relevance_vector[result_id] = 1
            query_dict[model] = relevance_vector
        relevance_dict[query_id] = query_dict
    return relevance_dict

relevance_dict = benchmarking(results_dict)

array_relevant = [13,11,12]


# Function which calculates average precision per query and model
def average_precision(arr, relevant):
    matrix = np.zeros([10,2])
    nominator = 0
    denominator = 0

    for index in range(len(arr)):
        denominator += 1
        if arr[index] == 1:
            nominator += 1
            matrix[index][0] = nominator/relevant
            matrix[index][1] = nominator/denominator
    
    avg_p = sum(matrix[:,1])/relevant
    return avg_p

# Function which gives MAP results for each model (based on the relevant documents presented in results which are pointed out by user)
def mean_average_precision(relevance_dict, array_relevant):
    n = len(relevance_dict.keys())
    model_list = ["vsm", "bm25", "mm"]
    mapositive_matrix = np.zeros([n, 3])
    for query_id in range(n):
        for model_id in range(3):
            mapositive_matrix[query_id][model_id] = average_precision(relevance_dict[query_id][model_list[model_id]], array_relevant[query_id])
    
    return np.apply_along_axis(np.mean, 0, mapositive_matrix)

# Function which gives a matrix with recall and precision values
def precision_recall_matrix(arr, relevant):
    matrix = np.zeros([10,2])
    nominator = 0
    denominator = 0

    for index in range(len(arr)):
        denominator += 1
        if arr[index] == 1:
            nominator += 1
            matrix[index][0] = nominator/relevant # Recall
            matrix[index][1] = nominator/denominator # Precision
        else:
            matrix[index][0] = nominator/relevant # Recal 
            matrix[index][1] = nominator/denominator

    return matrix

# Provides PR curves for each query comparing models
def pr_curve(relevance_dict, array_relevant):
    model_list = ["vsm", "bm25", "mm"]
    n = len(relevance_dict.keys())
    pr_matrix = {query_id: {model_id: 0 
                    for model_id in range(3)}
                for query_id in range(n)}

    for query_id in range(n):
        for model_id in range(3):
            pr_matrix[query_id][model_id] = precision_recall_matrix(relevance_dict[query_id][model_list[model_id]], array_relevant[query_id])

    fig, (ax1,ax2,ax3) = plt.subplots(3, figsize=(5,10))
    ax1.plot(pr_matrix[0][0][:,0], pr_matrix[0][0][:,1], '-o', 
                pr_matrix[0][1][:,0], pr_matrix[0][1][:,1], '-o', 
                pr_matrix[0][2][:,0], pr_matrix[0][2][:,1], '-o')
    ax1.set_xlim([-0.1, 1.1])
    ax1.set_ylim([-0.1, 1.1])
    ax1.legend(['Vector Space Model', 'BestMatch25 Model', "Mixture Model"])

    ax2.plot(pr_matrix[1][0][:,0], pr_matrix[1][0][:,1], '-o', 
                pr_matrix[1][1][:,0], pr_matrix[1][1][:,1], '-o', 
                pr_matrix[1][2][:,0], pr_matrix[1][2][:,1], '-o')
    ax2.set_xlim([-0.1, 1.1])                
    ax2.set_ylim([-0.1, 1.1])
    ax2.legend(['Vector Space Model', 'BestMatch25 Model', "Mixture Model"])

    ax3.plot(pr_matrix[2][0][:,0], pr_matrix[2][0][:,1], '-o', 
                pr_matrix[2][1][:,0], pr_matrix[2][1][:,1], '-o', 
                pr_matrix[2][2][:,0], pr_matrix[2][2][:,1], '-o')
    ax3.set_xlim([-0.1, 1.1])
    ax3.set_ylim([-0.1, 1.1])
    ax3.legend(['Vector Space Model', 'BestMatch25 Model', "Mixture Model"])

    return plt.show()

