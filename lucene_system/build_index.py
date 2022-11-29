import sys, lucene, os
import time
import pandas as pd

import traceback


from java.io import StringReader
from java.nio.file import Path, Paths
from org.apache.lucene.document import Document, Field, StoredField, StringField, TextField
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.analysis.standard import StandardTokenizer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.store import FSDirectory


def loadCorpus():
    print("Reading CSVs...")
    start = time.time()
    bbcDataFrame = pd.read_csv("../data/input_data/tweets_bbc.csv", usecols=["id", "tweet"])
    cnnDataFrame = pd.read_csv("../data/input_data/tweets_cnn.csv", usecols=["id", "tweet"])
    ecoDataFrame = pd.read_csv("../data/input_data/tweets_eco.csv", usecols=["id", "tweet"])
    print(f"Took {time.time()-start} seconds.\n")

    frames = [bbcDataFrame, cnnDataFrame, ecoDataFrame]
    return pd.concat(frames)


def getDoc(id, tweet):
    doc = Document()
    doc.add(Field("docid", id, StringField.TYPE_STORED))
    doc.add(Field("content", tweet, TextField.TYPE_STORED))

    return doc

def indexSingleLine(line, w):      #function used to exploit the pandas.agg functionality
    #takes in a row, creates new document object, and adds it to writer.
    document = getDoc(line["id"], line["tweet"])
    w.addDocument(document)


def indexCorpus(corpus):
    print("Creating corpus...")
    start = time.time()
    dirPath = Paths.get("./lucene_index")
    directory = FSDirectory.open(dirPath)
    analyzer = StandardAnalyzer()                   #do stopword removal etc. here (can also use different types of analysers) - https://lucene.apache.org/core/9_0_0/core/org/apache/lucene/analysis/standard/StandardAnalyzer.html
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


