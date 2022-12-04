from flask import Flask, render_template, request, redirect, url_for
from search_engine.search_models import load_data, vector_space_model, BM25, mixture_model
from search_engine.helpers import preprocess_string 
import pandas as pd

INPUT_DATAFRAME_PATH = "search_engine/model_components/processed_data.csv"
MODEL_COMPOMENTS_PATH = "search_engine/model_components/"


app = Flask(__name__)
loaded_data = load_data(MODEL_COMPOMENTS_PATH, INPUT_DATAFRAME_PATH)

@app.route('/')
@app.route('/home')
def index():
    return render_template('index.html')



@app.route('/search-engine', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form['query']
        model = request.form['model']
        n = int(request.form['n'])
        query = preprocess_string(query)
        if model == 'vsm':
            results = vector_space_model(
                    query, 
                    loaded_data['tweet_vectors'], 
                    loaded_data['idf_dict'], 
                    loaded_data['term_id'], 
                    loaded_data['all_tweets'], 
                    N=n
                )
        elif model == 'bm25':
            results = BM25(
                    query, 
                    loaded_data['factors_matrix'], 
                    loaded_data['idf_dict'], 
                    loaded_data['term_id'], 
                    loaded_data['all_tweets'], 
                    k_3=1.5, 
                    N=n
                )
        elif model == 'mixture':
            results = mixture_model(
                    query, 
                    loaded_data['parameter_matrix'], 
                    loaded_data['term_id'], 
                    loaded_data['all_tweets'], 
                    loaded_data['cf'], 
                    lam=0.5, 
                    N=n
        )

        # results = pd.DataFrame(results, columns=['tweet'])
        # return render_template('search_output.html', tables=[results.to_html(classes='data')], titles=results.columns.values)
        return render_template('search_output.html', results=results)
    else:
        return render_template('search_input.html')


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