import pandas as pd
import scipy.sparse as sp
import math
import time
from ast import literal_eval

rowIndex=0
OUTPUT_PATH = "data/indices/"


def loadData():
    print("Reading CSVs...")
    start = time.time()
    bbcDataFrame = pd.read_csv("data/input_data/tweets_bbc.csv", usecols=["tweet"])
    cnnDataFrame = pd.read_csv("data/input_data/tweets_cnn.csv", usecols=["tweet"])
    ecoDataFrame = pd.read_csv("data/input_data/tweets_eco.csv", usecols=["tweet"])
    print(f"Took {time.time()-start} seconds.\n")


    #StartIndex variables indicate the index that begins the section of tweets from that account.
    #i.e. bbc tweets start from row 0 until cnnStartIndex-1; cnn tweets are from cnnStartIndex until ecoStartIndex-1; economist tweets are from ecoStartIndex until the end.
    bbcStartIndex = 0
    cnnStartIndex = len(bbcDataFrame.index)
    ecoStartIndex = cnnStartIndex + len(cnnDataFrame.index)

    frames = [bbcDataFrame, cnnDataFrame, ecoDataFrame]
    allTweets = pd.concat(frames)
    tweetCount = len(allTweets.index)

    
def loadTokenized():
    print("Reading preprocessed CSV...")
    start = time.time()
    returnDF = pd.read_csv("data/processed_data/processed_data.csv")
    returnDF["preprocessed_text"] = list(map(lambda tweet : literal_eval(tweet),returnDF["preprocessed_text"])) #needed to convert string of array to plain array
    print(f"Took {time.time()-start} seconds.\n")
    return returnDF


def calcLogWeightedTF(tokenizedTweets):
    """
    calculate the Log Weighted Term Frequencies of terms in a series of processed tweets.

            Parameters:
                tokenizedTweets (pandas.Series): Series of arrays - each entry is an array representing a tokenized tweet.
            Returns:
                returnSeries (pandas.Series): Series of dictionaries - each series entry is a dictionary containing, for that row, the unique term:log-weighted TF score pairs.
    """
    #getting log weighted tf scores for each term
    print("Calculating weighted tf scores...")
    start = time.time()
    returnSeries = list(map(lambda tweet : dict(map(lambda token : (token, 1 + math.log(tweet.count(token), 10)), tweet)), tokenizedTweets))
    print(f"Took {time.time()-start} seconds.\n")
    return returnSeries

def calcUnweightedTF(tokenizedTweets):
    """
    calculate the Unweighted Term Frequencies of terms in a series of processed tweets.

            Parameters:
                tokenizedTweets (pandas.Series): Series of arrays - each entry is an array representing a tokenized tweet.
            Returns:
                returnSeries (pandas.Series): Series of dictionaries - each series entry is a dictionary containing, for that row, the unique term:TF score pairs.
    """
    print("Calculating tf scores...")
    start = time.time()
    returnSeries = list(map(lambda tweet : dict(map(lambda token : (token, tweet.count(token)), tweet)), tokenizedTweets))
    print(f"Took {time.time()-start} seconds.\n")
    return returnSeries

def calcDFScores(termFreqSeries):
    """
    calculate the Document Frequencies of terms from a series of term:TF dictionaries.

            Parameters:
                tokenizedTweets (pandas.Series): series of term:TF dictionaries.
            Returns:
                dfDict (dict): dictionary holding term:document frequency pairs for all terms in the corpus.
    """
    dfDict = {}

    def getDF(rowDict):
        for term in rowDict:
            if not term in dfDict:
                dfDict[term] = 1
            else:
                dfDict[term] += 1

    print("Accumulating df scores...")
    start = time.time()
    termFreqSeries.agg([getDF])
    print(f"Took {time.time()-start} seconds.\n")
    return dfDict


def calcIDFScores(dfDict, noOfDocs):
    """
    Calculates idf scores for each term, while also making a term order dictionary
    Term order dictionary stores term:(index in list) pairs to allow for reverese lookup of the index of a term in the model vector for faster production of vectors.
    (this means the program doesnt have to do a linear search every time it needs to know a term's position when creating the tf-idf matrix)
            Parameters:
                dfDict (dict): Dictionary containing term:df pairs for all terms in the corpus.
                noOfDocs (int): Total number of documents in the corpus
            Returns:
                idfDict (dict): Dictionary containing term:idf pairs for all terms in the corpus.
                termIndexDict (dict): Dictionary containing term:index pairs for fast reverse lookup.
    """
    print("Calculating idf scores...")
    start = time.time()
    idfDict = {}
    termIndexDict = {}
    termIndex = 0
    for term, df in dfDict.items():
        idfDict[term] = math.log(noOfDocs/df, 10)
        termIndexDict[term] = termIndex
        termIndex += 1

    print(f"Took {time.time()-start} seconds.\n")
    return idfDict, termIndexDict


def generateTweetVectors(tweetTFDicts, idfDict, termIndexDict):
    """
    Generates a tf-idf matrix that represents the corpus.
    Each row represents a tweet, each column represents a unique term from the entire corpus.
    Each entry is the tf-idf score of the corresponding word in the context of the corresponding tweet.
            Parameters:
                tweetTFDicts (pandas.Series): Series of dictionaries - each series entry is a dictionary containing, for that row, the unique term:log-weighted TF score pairs.
                idfDict (dict): Dictionary containing term:idf pairs for all terms in the corpus.
                termIndexDict (dict): Dictionary containing term:index pairs for fast reverse lookup.
            Returns:
                tweetVectors (scipy.sparse.dok_matrix): Sparse tf-idf matrix representation of the corpus.
    """
    termCount = len(idfDict)
    tweetCount = tweetTFDicts.size

    #generating document vectors
    print(f"Generating tweet vectors ({tweetCount} x {termCount} matrix)")
    start = time.time()
    tweetVectors = sp.dok_matrix((tweetCount, termCount), dtype="float64")


    #TODO: speed this up - maybe use pandarallel library to parallelize
    def populateVector(tweetTerms):
        global rowIndex
        for term, tfWeight in tweetTerms.items():
            tweetVectors[rowIndex, termIndexDict[term]] = tfWeight * idfDict[term]
        rowIndex += 1

    tweetTFDicts.agg([populateVector])          #agg marginally quicker than pandarallel (~30s vs ~33s)

    print(f"Took {time.time()-start} seconds.\n")

    return tweetVectors




def generateAllStructures():
    #load in tokenized data
    tweets = loadTokenized()

    #get log weighted tf scores
    tweets["tf"] = calcLogWeightedTF(tweets["preprocessed_text"])

    print("tf scores:")
    print(tweets["tf"])
    #get dictionary containing all terms in corpus and their document frequency.
    dfScores = calcDFScores(tweets["tf"])

    #calculate IDF scores, and create a term:position-in-list dictionary for quick reverse lookup.
    idfScores, termIndexDict = calcIDFScores(dfScores, len(tweets.index))

    #generate tf-idf matrix
    tweetVectors = generateTweetVectors(tweets["tf"], idfScores, termIndexDict)

    #outputting data to files.
    print("Saving data to files.")
    start = time.time()
    idfDataFrame = pd.DataFrame(idfScores.items(), columns=["term", "idf"])
    idfDataFrame.to_csv(OUTPUT_PATH+"idfDict.csv", encoding="utf-8", index=False)

    termindexDataFrame = pd.DataFrame(termIndexDict.items(), columns=["term", "index"])
    termindexDataFrame.to_csv(OUTPUT_PATH+"termIndexDict.csv", encoding="utf-8", index=False)

    sp.save_npz(OUTPUT_PATH+"documentVectors.npz", tweetVectors.tocoo())
    print(f"Took {time.time()-start} seconds.\n")

    return tweets, idfScores, termIndexDict, tweetVectors





if __name__ == "__main__":
    generateAllStructures()


