from functions.search_engine.custom_models import load_data
from functions.search_engine.custom_models import vector_space_model
from functions.search_engine.custom_models import BM25
from functions.search_engine.custom_models import mixture_model
from functions.preprocessing.helpers import preprocess_string


# PATHS
MODEL_COMPONENTS_FOLDER = "data/04_model_components/custom_components/"
INPUT_DATAFRAME_PATH = "data/03_processed_data/processed_data.csv"


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
    # print("\nVector space model results:")
    # print(vsm)
    
    bm25 = BM25(
            query_tokenized, 
            loaded_data['factors_matrix'], 
            loaded_data['idf_dict'], 
            loaded_data['term_id'], 
            loaded_data['all_tweets'], 
            k_3=1.5, 
            N=n
        )
    # print("\nBestMatch25 model results:")
    # print(bm25)

    mm = mixture_model(
            query_tokenized, 
            loaded_data['parameter_matrix'], 
            loaded_data['term_id'], 
            loaded_data['all_tweets'], 
            loaded_data['cf'], 
            lam=0.5, 
            N=n
        )
    # print("\nMixture model results:")
    # print(mm)

    # return vsm, bm25, mm
    return mm


if __name__ == "__main__":
    data = load_data(MODEL_COMPONENTS_FOLDER, INPUT_DATAFRAME_PATH)
    # query = input('Please enter your query: ')
    query = 'lewis hamilton'
    print(run_all_models(query, data))
