import sys, lucene, os
import time
import traceback
from nltk.corpus import stopwords

sys.path.append("../data_preprocessing")    #adding path at runtime to allow use of preprocessing functions
from helpers import remove_duplicate_tweets


from java.nio.file import Paths
from org.apache.lucene.document import Document, Field, StringField, TextField
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.analysis import CharArraySet
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.store import FSDirectory


JOINED_DATA_PATH = "../data/joined_data/joined_data.csv"
DUPLICATE_COL = 'tweet'

def loadCorpus():
    '''
    Loads the corpus into a file and removes duplicate tweets using helpers.remove_duplicate_tweets function.
    Then drops unneeded columns.
    
            Returns:
                dataframe (pandas.DataFrame): data frame with 2 columns ("id" and "tweet") that contains each unique tweet.
    '''

    print("Reading CSVs, removing duplicates...")
    start = time.time()
    dataFrame = remove_duplicate_tweets(JOINED_DATA_PATH, DUPLICATE_COL)

    print(f"Took {time.time()-start} seconds.\n")
    return dataFrame[["id","tweet"]]


def _getStopwords():
    '''
    Function needed as cannot make a direct cast from python list to Java collection so this function iteratively builds the collection instead.
    Using the NLTK set of stopwords with the StandardAnalyzer yields better results than using the Lucene EnglishAnalyzer.
    (e.g. query of "it" will return empty set when using the NLTK set)

            Returns:
                javaSet (java type - org.apache.lucene.analysis.CharArraySet): CharArraySet containing all stopwords for use in a Lucene Analyzer.
    '''

    pythonSet = stopwords.words("english")
    javaSet = CharArraySet(len(pythonSet), True)
    for word in pythonSet:
        javaSet.add(word)

    return javaSet
    


def _createDoc(id, tweet):
    '''
    Creates and returns a lucene document from the input.
    
            Parameters:
                id (string): Unique ID of the tweet being passed.
                tweet (string): The tweet that will be indexed and used to compare queries to.
                
                Returns:
                    doc (java type - org.apache.lucene.document): Lucene encoded document that can be added to an index.
    '''

    doc = Document()
    doc.add(Field("id", id, StringField.TYPE_STORED))
    doc.add(Field("tweet", tweet, TextField.TYPE_STORED))

    return doc


def _indexSingleLine(line, w):
    '''
    Function used to exploit the pandas.agg functionality.
    Takes in a row, creates new document object, and adds it to writer.

            Parameters:
                line (pandas.DataFrame): A single row from the pandas dataframe
                w (java type - org.apache.lucene.index.IndexWriter): Writer object that adds document to index.
    '''

    document = _createDoc(line["id"], line["tweet"])
    w.addDocument(document)


def indexCorpus(corpus):
    '''
    Builds an index from the given corpus and writes the index to a folder in the same directory.
    
            Parameters:
                corpus (pandas.DataFrame): Dataframe containing "id" and "tweet" columns corresponding to unique ID of a tweet, and the tweet itself respectively.
    '''

    print("Creating corpus...")
    indexDirName = "./lucene_index"
    start = time.time()
    if not os.path.exists("./lucene_index"):
        os.makedirs("./lucene_index")

    dirPath = Paths.get(indexDirName)
    directory = FSDirectory.open(dirPath)
    stopwordSet = _getStopwords()
    analyzer = StandardAnalyzer(stopwordSet)
    config = IndexWriterConfig(analyzer)
    config.setOpenMode(IndexWriterConfig.OpenMode.CREATE_OR_APPEND)

    try:
        writer = IndexWriter(directory, config)
        corpus.agg(_indexSingleLine, axis=1, w=writer)
    
    except Exception:
        print("there was a problem")
        traceback.print_exc()
        
    finally:
        writer.close()

    print("Successfully built index.")
    print(f"Took {time.time()-start} seconds.\n")





if __name__ == "__main__":
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    tweetsDF = loadCorpus()
    #print(tweetsDF)
    indexCorpus(tweetsDF)


