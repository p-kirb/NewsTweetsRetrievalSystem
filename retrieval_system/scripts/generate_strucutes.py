from functions.preprocessing.create_components import loadTokenized
from functions.preprocessing.create_components import calcLogWeightedTF
from functions.preprocessing.create_components import calcUnweightedTF
from functions.preprocessing.create_components import calcDFScores
from functions.preprocessing.create_components import calcIDFScores
from functions.preprocessing.create_components import generateTweetVectors
from functions.preprocessing.create_components import factoring
from functions.preprocessing.create_components import parameter_estimation
from functions.preprocessing.create_components import CFt
import pandas as pd
import time
import scipy.sparse as sp


# PATHS
COMPONENTS_OUTPUT_FOLDER = "data/04_model_components/custom_components/"
INPUT_DATAFRAME_PATH = "data/03_processed_data/processed_data.csv"


def generateAllStructures(input_path, output_path):
    # load in tokenized data
    tweets = loadTokenized(input_path)

    # get log weighted tf scores
    tweets["weighted_tf"] = calcLogWeightedTF(tweets["preprocessed_text"])
    tweets["unweighted_tf"] = calcUnweightedTF(tweets["preprocessed_text"])

    # get dictionary containing all terms in corpus and their document frequency.
    dfScores = calcDFScores(tweets["weighted_tf"])

    # calculate IDF scores, and create a term:position-in-list dictionary for quick reverse lookup.
    idfScores, termIndexDict = calcIDFScores(dfScores, len(tweets.index))

    # generate tf-idf matrix
    tweetVectors = generateTweetVectors(tweets["weighted_tf"], idfScores, termIndexDict)

    # Calculate factor matrix (for BM25 algorithm)
    factors_matrix = factoring(tweets["unweighted_tf"], idfScores, termIndexDict, k=1.5, b=0.75)
    
    # Calculate parameter matrix (for mixture model)
    parameter_matrix = parameter_estimation(tweets["unweighted_tf"], idfScores, termIndexDict)

    # Calculate cf (for mixture model)
    cf = CFt(tweets["unweighted_tf"])

    # Saving data to files.
    print("Saving data to files.")

    start = time.time()

    # Save tweet vectors as csv
    idfDataFrame = pd.DataFrame(idfScores.items(), columns=["term", "idf"])
    idfDataFrame.to_csv(output_path+"idfDict.csv", encoding="utf-8", index=False)

    # Save termIndexDict as csv
    termindexDataFrame = pd.DataFrame(termIndexDict.items(), columns=["term", "index"])
    termindexDataFrame.to_csv(output_path+"termIndexDict.csv", encoding="utf-8", index=False)

    # Save tweet vectors as sparse matrix
    sp.save_npz(output_path+"documentVectors.npz", tweetVectors.tocoo())

    # Save factors matrix as sparse matrix
    sp.save_npz(output_path+"factorsMatrix.npz", factors_matrix.tocoo())

    # Save parameter matrix as sparse matrix
    sp.save_npz(output_path+"parameterMatrix.npz", parameter_matrix.tocoo())

    # Save cf as csv
    cfDataFrame = pd.DataFrame(cf.items(), columns=["term", "cf"])
    cfDataFrame.to_csv(output_path+"cfDict.csv", encoding="utf-8", index=False)

    print(f"Took {time.time()-start} seconds.\n")


if __name__ == "__main__":
    generateAllStructures(INPUT_DATAFRAME_PATH, COMPONENTS_OUTPUT_FOLDER)