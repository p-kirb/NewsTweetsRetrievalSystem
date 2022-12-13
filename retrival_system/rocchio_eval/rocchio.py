import numpy as np
# import scipy.sparse as sp
# from search_models import vector_space_model, load_data
# from helpers import preprocess_string
# from itertools import compress
# from sklearn.preprocessing import normalize



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


# loded_data = load_data('model_components/', 'model_components/processed_data.csv')

# print(loded_data['all_tweets'].head())

# query = 'el chapo escapes prison'

# query_tokenized = preprocess_string(query)

# print('QT: ', query_tokenized)

# results, old_query \
#     = vector_space_model(
#                     query_tokenized, 
#                     loded_data['tweet_vectors'], 
#                     loded_data['idf_dict'], 
#                     loded_data['term_id'], 
#                     loded_data['all_tweets'], 
#                     N=15
#                 )

# print('VS: ', results)

# for t in loded_data['all_tweets']['tweet'].iloc[results.index]:
#     print(t, '\n')

# relevant = [1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0]
# relevant_docs_ids = list(compress(results.index, relevant))
# nonrelevant_docs_ids = list(compress(results.index, [not i for i in relevant]))

# print(relevant_docs_ids)
# print(loded_data['tweet_vectors'])

# tf_idf = loded_data['tweet_vectors'].tocsr()

# rel_docs = tf_idf[relevant_docs_ids, :]
# nonrel_docs = tf_idf[nonrelevant_docs_ids, :]

# new_query = rocchio_algorithm(old_query, rel_docs, nonrel_docs)


# tweetVectors = normalize(tf_idf, norm='l2', axis = 1)

# cosine_similarity = tweetVectors.dot(new_query)
# doc_id = np.argsort(-cosine_similarity)[0:15]
# results2 = loded_data['all_tweets']["tweet"][doc_id]

# for old, new in zip(loded_data['all_tweets']['tweet'].iloc[results.index], results2):
#     print(old)
#     print(new, '\n')