from functions.evaluation.evaluation_static import *
from functions.search_engine.custom_models import load_data

TWEETS_DATAFRAME_PATH = "data/03_processed_data/processed_data.csv"
MODEL_COMPOMENTS_PATH = "data/04_model_components/custom_components/"
RELEVANT_QUERIES_PATH = "data/05_evaluation/relevant_queries.csv"
ROCCHIO_PATH = "data/05_evaluation/rocchio_results.json"
PLOTS_OUTPUT_PATH = "data/05_evaluation/plots/"

q1 = "Thailand cave rescue"
q2 = "Notre-dame fire"
q3 = "US capitol building storming"
q4 = "water found on Mars"
q5 = "El Chapo escapes prison"
queries = [q1, q2, q3, q4, q5]


# Run all components
def run_all(queries, plots_output_path):
    # Load data and components
    loaded_data = load_data(MODEL_COMPOMENTS_PATH, TWEETS_DATAFRAME_PATH)
    # Load sets of relevant queries
    relevant_queries = load_relevant_queries(RELEVANT_QUERIES_PATH)
    # Get ids lists of relevant query and their totals
    relevant_doc_dict, total_relevant = get_tweet_id(queries, loaded_data, relevant_queries)
    # Load Rocchio results
    rocchio_results = load_rocchio_results(ROCCHIO_PATH)
    # Run all models
    results_dict = running_results(queries, loaded_data, total_relevant, rocchio_results)
    # Compare results with the results sets
    relevance_vector = benchmarking(results_dict, relevant_doc_dict, queries)
    # Calculate MAP results for each model
    map_results = mean_average_precision(relevance_vector, total_relevant, queries)
    # Plot PR-curves 
    fig, ax = pr_curve(relevance_vector, total_relevant, queries, plots_output_path)

    return map_results


if __name__ == "__main__":
    map_results = run_all(queries, PLOTS_OUTPUT_PATH)
    print(map_results)