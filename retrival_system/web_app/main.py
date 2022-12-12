from flask import Flask, render_template, request, redirect, url_for
from search_engine.search_models import load_data, vector_space_model, BM25, mixture_model
from search_engine.helpers import preprocess_string 
from search_engine.rocchio import rocchio_algorithm
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd

INPUT_DATAFRAME_PATH = "search_engine/model_components/processed_data.csv"
MODEL_COMPOMENTS_PATH = "search_engine/model_components/"


app = Flask(__name__)
loaded_data = load_data(MODEL_COMPOMENTS_PATH, INPUT_DATAFRAME_PATH)

@app.route('/')
@app.route('/home')
def index():
    return render_template('index.html')


@app.route('/rocchio', methods=['POST'])
def rocchio():
    query = request.form['query']
    model = request.form['model']
    n = int(request.form['n'])

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
    doc_id = np.argsort(-cosine_similarity)[0:n]
    new_results = loaded_data['all_tweets']["id"][doc_id]

    return render_template('search_input.html', results=list(new_results), query=query, model=model, n=n)


@app.route('/search-engine', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form['query']
        model = request.form['model']
        n = int(request.form['n'])
        processed_query = preprocess_string(query)
        if model == 'vsm':
            results, tf_idf = vector_space_model(
                    processed_query, 
                    loaded_data['tweet_vectors'], 
                    loaded_data['idf_dict'], 
                    loaded_data['term_id'], 
                    loaded_data['all_tweets'], 
                    N=n
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
                    N=n
                )
        elif model == 'mixture':
            results = mixture_model(
                    processed_query, 
                    loaded_data['parameter_matrix'], 
                    loaded_data['term_id'], 
                    loaded_data['all_tweets'], 
                    loaded_data['cf'], 
                    lam=0.6, 
                    N=n
        )

        return render_template('search_input.html', results=results, query=query, model=model, n=n)
    else:
        return render_template('search_input.html', results=[], query='', model='bm25', n=10)


@app.route('/vector-space-model')
@app.route('/VSM')
def vsm():
    return render_template('vsm.html')


@app.route('/BM25')
def bm25():
    return render_template('bm25.html')


@app.route('/mixture-model')
def mixture():
    return render_template('mixture.html')


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)