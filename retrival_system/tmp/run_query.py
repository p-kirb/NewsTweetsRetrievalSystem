from importlib import import_module
from search_models import *


INPUT_DATA_PATH = "data/model_compoments/"

# MAIN FUNCTION
def run_all_models(query, loaded_data, n=10):
    # Query tokenization
    query_tokenized = preprocess_string(query)
    
    # Modelling
    vsm = vector_space_model(
            query_tokenized, 
            loaded_data['tweet_vectors'], 
            loaded_data['idf_dict'], 
            loaded_data['term_id'], 
            loaded_data['all_tweets'], 
            N=n
        )
    print("\nVector space model results:")
    print(vsm)
    
    bm25 = BM25(
            query_tokenized, 
            loaded_data['factors_matrix'], 
            loaded_data['idf_dict'], 
            loaded_data['term_id'], 
            loaded_data['all_tweets'], 
            k_3=1.5, 
            N=n
        )
    print("\nBestMatch25 model results:")
    print(bm25)

    mm = mixture_model(
            query_tokenized, 
            loaded_data['parameter_matrix'], 
            loaded_data['term_id'], 
            loaded_data['all_tweets'], 
            loaded_data['cf'], 
            lam=0.5, 
            N=n
        )
    print("\nMixture model results:")
    print(mm)

    return vsm, bm25, mm


if __name__ == "__main__":
    data = load_data(INPUT_DATA_PATH)
    query = input('Please enter your query: ')
    run_all_models(query, data)