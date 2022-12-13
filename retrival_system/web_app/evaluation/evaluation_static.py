import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from search_engine.search_models import load_data, vector_space_model, BM25, mixture_model
from search_engine.helpers import preprocess_string, remove_duplicate_tweets
import pandas as pd

TWEETS_DATAFRAME_PATH = "search_engine/processed_data.csv"
MODEL_COMPOMENTS_PATH = "search_engine/model_components/"
RELEVANT_QUERIES_PATH = "evaluation/relevant_queries.xlsx"

q1 = "Thailand cave rescue"
q2 = "Notre-dame fire"
q3 = "US capitol building storming"
q4 = "water found on Mars"
q5 = "El Chapo escapes prison"
queries = [q1, q2, q3, q4, q5]


# Function which tweets ids to relevant documents
def get_tweet_id(queries, loaded_data, relevant_queries):
    relevant_dict = {}
    total_relevant = {}
    for query in queries:
        relevant_set = relevant_queries[query].dropna()
        relevant_dict[query] = list(map(lambda x: 
                                                int(loaded_data["all_tweets"]["id"][loaded_data["all_tweets"]["tweet"] == x])
                                                ,relevant_set))
        total_relevant[query] = len(relevant_dict[query])
    return relevant_dict, total_relevant

# Function which loads sets of relevant documents
def load_relevant_queries(RELEVANT_QUERIES_PATH):
    relevant_queries = pd.read_excel(RELEVANT_QUERIES_PATH)
    return relevant_queries

# Function which runs results from all models
def running_results(queries, loaded_data, total_relevant):
    results_dict = {}
    for query in queries:
        n = total_relevant[query]
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
                    lam=0.6, 
                    N=n)

        rocchio_results = rocchio_data[query]

        results_dict[query] = {"vsm": vsm_results, "bm25": bm25_results, "mm": mm_results, "rocchio": rocchio_results}

    return results_dict

# Function which compares system's results with relevance sets
def benchmarking(results_dict, relevant_dict):
    relevance_dict = {}
    for query in queries:
        n = len(results_dict[query]["vsm"])
        relevant_set = relevant_dict[query]
        query_dict = {"vsm": 0, "bm25": 0, "mm": 0, "rocchio": 0}

        for model in query_dict.keys():
            relevance_vector = np.zeros(n)
            
            for result_id in range(n):
                if results_dict[query][model].iloc[result_id] in relevant_set:
                    relevance_vector[result_id] = 1

            query_dict[model] = relevance_vector
        relevance_dict[query] = query_dict

    return relevance_dict

# Function which calculates average precision per query and model
def average_precision(arr, relevant):
    matrix = np.zeros([len(arr),2])
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
def mean_average_precision(relevance_vector, total_relevant, queries):
    model_list = ["vsm", "bm25", "mm", "rocchio"]
    positive_matrix = np.zeros([len(queries), 4])
    row_counter = 0
    for query in queries:
        for model_id in range(4):
            positive_matrix[row_counter][model_id] = average_precision(relevance_vector[query][model_list[model_id]], total_relevant[query])
        row_counter += 1
    return np.apply_along_axis(np.mean, 0, positive_matrix)

# Function which gives a matrix with recall and precision values
def precision_recall_matrix(arr, relevant):
    matrix = np.zeros([len(arr),2])
    nominator = 0
    denominator = 0

    for index in range(len(arr)):
        denominator += 1
        if arr[index] == 1:
            nominator += 1
        matrix[index][0] = nominator/relevant # Recall
        matrix[index][1] = nominator/denominator # Precision

    return matrix

# Provides PR curves for each query comparing models
def pr_curve(relevance_vector, total_relevant, queries):
    model_list = ["vsm", "bm25", "mm", "rocchio"]
    pr_matrix = {query: {model: 0 
                    for model in range(len(model_list))}
                for query in range(len(queries))}
    row_counter = 0
    for query_id in range(len(queries)):
        for model_id in range(4):
            pr_matrix[query_id][model_id] = precision_recall_matrix(relevance_vector[queries[query_id]][model_list[model_id]], total_relevant[queries[query_id]])
        row_counter += 1

    fig, ax = plt.subplots(3,2)
    ax[0,0].plot(pr_matrix[0][0][:,0], pr_matrix[0][0][:,1], '-o', 
                pr_matrix[0][1][:,0], pr_matrix[0][1][:,1], '-o', 
                pr_matrix[0][2][:,0], pr_matrix[0][2][:,1], '-o',
                pr_matrix[0][3][:,0], pr_matrix[0][3][:,1], '-o')
    ax[0,0].set_xlim([-0.1, 1.1])
    ax[0,0].set_ylim([-0.1, 1.1])
    ax[0,0].legend(['Vector Space Model', 'BestMatch25 Model', "Mixture Model", "Rocchio"])
    ax[0,0].set_title(q1)
    ax[0,0].set_xlabel("Recall")
    ax[0,0].set_ylabel("Precision")

    ax[0,1].plot(pr_matrix[1][0][:,0], pr_matrix[1][0][:,1], '-o', 
                pr_matrix[1][1][:,0], pr_matrix[1][1][:,1], '-o', 
                pr_matrix[1][2][:,0], pr_matrix[1][2][:,1], '-o',
                pr_matrix[1][3][:,0], pr_matrix[1][3][:,1], '-o')
    ax[0,1].set_xlim([-0.1, 1.1])                
    ax[0,1].set_ylim([-0.1, 1.1])
    ax[0,1].legend(['Vector Space Model', 'BestMatch25 Model', "Mixture Model", "Rocchio"])
    ax[0,1].set_title(q2)
    ax[0,1].set_xlabel("Recall")
    ax[0,1].set_ylabel("Precision")

    ax[1,0].plot(pr_matrix[2][0][:,0], pr_matrix[2][0][:,1], '-o', 
                pr_matrix[2][1][:,0], pr_matrix[2][1][:,1], '-o', 
                pr_matrix[2][2][:,0], pr_matrix[2][2][:,1], '-o',
                pr_matrix[2][3][:,0], pr_matrix[2][3][:,1], '-o')
    ax[1,0].set_xlim([-0.1, 1.1])
    ax[1,0].set_ylim([-0.1, 1.1])
    ax[1,0].legend(['Vector Space Model', 'BestMatch25 Model', "Mixture Model", "Rocchio"])
    ax[1,0].set_title(q3)
    ax[1,0].set_xlabel("Recall")
    ax[1,0].set_ylabel("Precision")

    ax[1,1].plot(pr_matrix[3][0][:,0], pr_matrix[3][0][:,1], '-o', 
            pr_matrix[3][1][:,0], pr_matrix[3][1][:,1], '-o', 
            pr_matrix[3][2][:,0], pr_matrix[3][2][:,1], '-o',
            pr_matrix[3][3][:,0], pr_matrix[3][3][:,1], '-o')
    ax[1,1].set_xlim([-0.1, 1.1])
    ax[1,1].set_ylim([-0.1, 1.1])
    ax[1,1].legend(['Vector Space Model', 'BestMatch25 Model', "Mixture Model", "Rocchio"])
    ax[1,1].set_title(q4)
    ax[1,1].set_xlabel("Recall")
    ax[1,1].set_ylabel("Precision")

    ax[2,0].plot(pr_matrix[4][0][:,0], pr_matrix[4][0][:,1], '-o', 
            pr_matrix[4][1][:,0], pr_matrix[4][1][:,1], '-o', 
            pr_matrix[4][2][:,0], pr_matrix[4][2][:,1], '-o',
            pr_matrix[4][3][:,0], pr_matrix[4][3][:,1], '-o')
    ax[2,0].set_xlim([-0.1, 1.1])
    ax[2,0].set_ylim([-0.1, 1.1])
    ax[2,0].legend(['Vector Space Model', 'BestMatch25 Model', "Mixture Model", "Rocchio"])
    ax[2,0].set_title(q5)
    ax[2,0].set_xlabel("Recall")
    ax[2,0].set_ylabel("Precision")

    return plt.show()

# Run all components
def run_all(queries):
    # Load data:
    loaded_data = load_data(MODEL_COMPOMENTS_PATH, TWEETS_DATAFRAME_PATH)
    # Load sets of relevant queries
    relevant_queries = load_relevant_queries(RELEVANT_QUERIES_PATH)
    # Get ids lists of relevant query and their totals
    relevant_doc_dict, total_relevant = get_tweet_id(queries, loaded_data, relevant_queries)
    # Run all models
    results_dict = running_results(queries, loaded_data, total_relevant)
    # Compare results with the results sets
    relevance_vector = benchmarking(results_dict, relevant_doc_dict)
    # Calculate MAP results for each model
    map_results = mean_average_precision(relevance_vector, total_relevant, queries)
    # Plot PR-curves 
    pr_plots = pr_curve(relevance_vector, total_relevant, queries)

    return map_results


