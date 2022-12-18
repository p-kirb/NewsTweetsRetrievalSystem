from flask import Flask, render_template, request, flash
from functions.search_engine.custom_models import load_data, vector_space_model, BM25, mixture_model
from functions.preprocessing.helpers import preprocess_string 
from functions.search_engine.rocchio import rocchio_algorithm
from functions.search_engine.lucene_model import lucene_query
import lucene
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd


# PATHS
MODEL_COMPOMENTS_PATH = "../data/04_model_components/custom_components/"
LUCENE_COMPONENTS_PATH = "../data/04_model_components/lucene_components/"
INPUT_DATAFRAME_PATH = "../data/03_processed_data/processed_data.csv"


# FLASK APP
app = Flask(__name__, template_folder='templates', static_folder='styles')
app.secret_key = 'hello there!'
loaded_data = load_data(MODEL_COMPOMENTS_PATH, INPUT_DATAFRAME_PATH)


# INITIALIZE LUCENE
@app.before_first_request
def initialize():
    global vm_env
    vm_env = lucene.initVM()


@app.route('/')
@app.route('/home')
@app.route('/search-engine', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form['query']
        model = request.form['model']
        n_to_show = int(request.form['n_to_show'])
        processed_query = preprocess_string(query)
        if model == 'vsm':
            results, tf_idf = vector_space_model(
                    processed_query, 
                    loaded_data['tweet_vectors'], 
                    loaded_data['idf_dict'], 
                    loaded_data['term_id'], 
                    loaded_data['all_tweets'], 
                    N=n_to_show
                )
            loaded_data['results'] = results
            loaded_data['tf_idf'] = tf_idf

        elif model == 'bm25':
            results = BM25(
                    processed_query, 
                    loaded_data['factors_matrix'], 
                    loaded_data['idf_dict'], 
                    loaded_data['term_id'], 
                    loaded_data['all_tweets'], 
                    k_3=1.5, 
                    N=n_to_show
                )
        elif model == 'mixture':
            results = mixture_model(
                    processed_query, 
                    loaded_data['parameter_matrix'], 
                    loaded_data['term_id'], 
                    loaded_data['all_tweets'], 
                    loaded_data['cf'], 
                    lam=0.6, 
                    N=n_to_show
        )
        elif model == 'lucene':
            if query == '':
                flash('Please enter a query')
                return render_template('search_engine.html', results=[], query='', model='lucene', n_to_show=10, n_results=0)
            
            vm_env.attachCurrentThread()
            results = lucene_query(query, n_to_show, LUCENE_COMPONENTS_PATH)

            results = results['id']


        return render_template('search_engine.html', results=results, query=query, model=model, n_to_show=n_to_show, n_results=len(results))
    else:
        return render_template('search_engine.html', results=[], query='', model='bm25', n_to_show=10, n_results=0)


@app.route('/rocchio', methods=['POST'])
def rocchio():
    query = request.form['query']
    model = request.form['model']
    n_to_show = int(request.form['n_to_show'])

    old_tf_idf = loaded_data['tf_idf']
    results = pd.DataFrame(loaded_data['results'])    

    relevant = [int(i) for i in request.form.getlist('relevant')]
    relevant_ids = results[results['id'].isin(relevant)].index
    nonrelevant_ids = results[~results['id'].isin(relevant)].index
    tf_idf = loaded_data['tweet_vectors'].tocsr()
    rel_docs = tf_idf[relevant_ids, :]
    nonrel_docs = tf_idf[nonrelevant_ids, :]
    new_query = rocchio_algorithm(old_tf_idf, rel_docs, nonrel_docs)

    tweetVectors = normalize(tf_idf, norm='l2', axis = 1)
    cosine_similarity = tweetVectors.dot(new_query)
    doc_id = np.argsort(-cosine_similarity)[0 : n_to_show]
    new_results = loaded_data['all_tweets']["id"][doc_id]

    return render_template('search_engine.html', results=list(new_results), query=query, model=model, n_to_show=n_to_show, n_results=len(new_results))


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)