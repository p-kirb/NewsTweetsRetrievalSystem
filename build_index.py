import pandas as pd
import scipy.sparse as sp
import math
import time


print("Reading CSVs...")
start = time.time()
bbcDataFrame = pd.read_csv("../datasets/tweets_bbc.csv", usecols=["tweet"])
cnnDataFrame = pd.read_csv("../datasets/tweets_cnn.csv", usecols=["tweet"])
ecoDataFrame = pd.read_csv("../datasets/tweets_eco.csv", usecols=["tweet"])
print(f"Took {time.time()-start} seconds.\n")


#StartIndex variables indicate the index that begins the section of tweets from that account.
#i.e. bbc tweets start from row 0 until cnnStartIndex-1; cnn tweets are from cnnStartIndex until ecoStartIndex-1; economist tweets are from ecoStartIndex until the end.
bbcStartIndex = 0
cnnStartIndex = len(bbcDataFrame.index)
ecoStartIndex = cnnStartIndex + len(cnnDataFrame.index)

frames = [bbcDataFrame, cnnDataFrame, ecoDataFrame]
allTweets = pd.concat(frames)
tweetCount = len(allTweets.index)


#INSERT FULL PREPROCESSING STEP HERE
print("Tokenising corpus...")
start = time.time()
allTweets["tokenized"] = list(map(lambda x: x.split(" "), allTweets["tweet"]))
print(f"Took {time.time()-start} seconds.\n")


#getting log weighted tf scores for each term
print("Calculating tf scores...")
start = time.time()
allTweets["tf"] = list(map(lambda tweet : dict(map(lambda token : (token, 1 + math.log(tweet.count(token), 10)), tweet)), allTweets["tokenized"]))
print(f"Took {time.time()-start} seconds.\n")


#getting df scores for each term
dfDict = {}

def getDF(rowDict):
    for term in rowDict:
        if not term in dfDict:
            dfDict[term] = 1
        else:
            dfDict[term] += 1

print("Accumulating df scores...")
start = time.time()
allTweets["tf"].agg([getDF])
print(f"Took {time.time()-start} seconds.\n")



#calculating idf scores for each term, also making a term order dictionary
#term index dictionary stores term:(index in list) pairs to allow for lookup of index of term in vector for faster production of vectors.
#(this means the program doesnt have to do a linear search every time it needs to know a term's position)
print("Calculating idf scores...")
def calcIDF(df):
    return math.log(tweetCount/df)

start = time.time()
idfDict = {}
termIndexDict = {}
termIndex = 0
for term, df in dfDict.items():
    idfDict[term] = calcIDF(df)
    termIndexDict[term] = termIndex
    termIndex += 1

print(f"Took {time.time()-start} seconds.\n")


termCount = len(idfDict)

#generating document vectors
print(f"Generating tweet vectors ({tweetCount} x {termCount} matrix)")
start = time.time()
tweetVectors = sp.dok_matrix((tweetCount, termCount), dtype="float64")


#TODO: speed this up - maybe use pandarallel library to parallelize
rowIndex=0
def populateVector(tweetTerms):
    global rowIndex
    for term, tfWeight in tweetTerms.items():
        tweetVectors[rowIndex, termIndexDict[term]] = tfWeight * idfDict[term]
    rowIndex += 1

allTweets["tf"].agg([populateVector])

print(f"Took {time.time()-start} seconds.\n")







#outputting data to files.
print("Saving data to files.")
start = time.time()
idfDataFrame = pd.DataFrame(idfDict.items(), columns=["term", "idf"])
idfDataFrame.to_csv("idfDict.csv", encoding="utf-8", index=False)

termindexDataFrame = pd.DataFrame(termIndexDict.items(), columns=["term", "index"])
termindexDataFrame.to_csv("termIndexDict.csv", encoding="utf-8", index=False)

allTweets.to_csv("tokenizedTweets.csv", encoding="utf-8", index=False)
sp.save_npz("documentVectors.npz", tweetVectors.tocoo())
print(f"Took {time.time()-start} seconds.\n")

