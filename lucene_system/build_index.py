import sys, lucene, os
import time
import pandas as pd
import traceback
from nltk.corpus import stopwords

sys.path.append("../data_preprocessing")    #adding path at runtime to allow use of preprocessing functions
from helpers import remove_duplicate_tweets


from java.io import StringReader
from java.nio.file import Path, Paths
from org.apache.lucene.document import Document, Field, StoredField, StringField, TextField
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.analysis import CharArraySet
from org.apache.lucene.analysis.standard import StandardTokenizer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.store import FSDirectory


JOINED_DATA_PATH = "../data/joined_data/joined_data.csv"
DUPLICATE_COL = 'tweet'

def loadCorpus():
    print("Reading CSVs, removing duplicates...")
    start = time.time()
    dataFrame = remove_duplicate_tweets(JOINED_DATA_PATH, DUPLICATE_COL)


    print(f"Took {time.time()-start} seconds.\n")
    return dataFrame[["id","tweet"]]


def getStopwords():
    '''
        function needed as cannot directly cast from python list to Java collection
        so instead, iteratively builds the collection.
        Using the NLTK set of stopwords with the StandardAnalyzer yields better results
        than using the Lucene EnglishAnalyzer. (e.g. query of "it" will return empty set
        when using the NLTK set)
    '''
    pythonSet = stopwords.words("english")
    javaSet = CharArraySet(len(pythonSet), True)
    for word in pythonSet:
        javaSet.add(word)

    return javaSet
    


def getDoc(id, tweet):
    doc = Document()
    doc.add(Field("docid", id, StringField.TYPE_STORED))
    doc.add(Field("content", tweet, TextField.TYPE_STORED))

    return doc

def indexSingleLine(line, w):
    '''
        function used to exploit the pandas.agg functionality
        takes in a row, creates new document object, and adds it to writer.
    '''
    document = getDoc(line["id"], line["tweet"])
    w.addDocument(document)


def indexCorpus(corpus):
    print("Creating corpus...")
    start = time.time()
    dirPath = Paths.get("./lucene_index")
    directory = FSDirectory.open(dirPath)
    stopwordSet = getStopwords()
    analyzer = StandardAnalyzer(stopwordSet)
    config = IndexWriterConfig(analyzer)
    config.setOpenMode(IndexWriterConfig.OpenMode.CREATE_OR_APPEND)

    try:
        writer = IndexWriter(directory, config)


        corpus.agg(indexSingleLine, axis=1, w=writer)
    
    except Exception:
        print("there was a problem")
        traceback.print_exc()
        
    finally:
        writer.close()



    #document = getDoc("502", "this is an example tweet")
    #writer.addDocument(document)

    #document2 = getDoc("502", "this is another example tweet")
    #writer.addDocument(document2)
    print(f"Took {time.time()-start} seconds.\n")





if __name__ == "__main__":
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    tweetsDF = loadCorpus()
    print(tweetsDF)
    indexCorpus(tweetsDF)


