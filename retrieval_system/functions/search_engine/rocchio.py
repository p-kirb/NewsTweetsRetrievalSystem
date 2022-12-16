import numpy as np


def rocchio_algorithm(query_tf_idf, rel_docs, nonrel_docs):
    '''
    Aims to calculate the new query vector using Rocchio algorithm.
            Parameters:
                query_tf_idf (array): Array containing tf-idf vector of the query. 
                rel_docs (array): Array containing tf-idf vector of the relevant documents.
                nonrel_docs (array): Array containing tf-idf vector of the non-relevant documents.
            Returns:
                new_query (array): Array containing the new query vector.   
                
    '''
    alpha = 1
    beta = 0.75
    gamma = 0.25
    new_query \
        = \
            alpha * query_tf_idf.reshape(1, -1) \
            + (beta * np.mean(rel_docs, axis = 0)).reshape(-1) \
            - (gamma * np.mean(nonrel_docs, axis = 0)).reshape(-1)

    return np.array(new_query)[0, :]