import sys, os, lucene
import pandas as pd

from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.search import IndexSearcher


def lucene_query(queryString, noOfResults, printResults=False):
    '''
    Makes a query to the Lucene index based on given query string.
    
            Parameters:
                queryString (string): Query used to retrieve documents.
                noOfResults (int): Upper bound of results that should be returned by query.
                printResults (boolean): function prints results if true.

                Returns:
                    resultsDF (pandas.DataFrame): Dataframe with columns "score", "id", and "tweet" containing the results.
    '''
    
    print(f"Retrieving top {noOfResults} results for query: '{queryString}'")

    dirPath = Paths.get("./lucene_index")
    directory = FSDirectory.open(dirPath)
    analyzer = StandardAnalyzer()                   #removal of stopwords in query is unneccesary as they are not indexed.
    
    resultsList = []

    try:
        searcher = IndexSearcher(DirectoryReader.open(directory))
        query = QueryParser("tweet", analyzer).parse(queryString)
        topDocs = searcher.search(query, noOfResults)
        print(f"Got {topDocs.totalHits.value} results:")
        for result in topDocs.scoreDocs:
            ranking = result.score
            tweet = searcher.doc(result.doc).get("tweet")
            id = searcher.doc(result.doc).get("id")
            resultsList.append([ranking, id, tweet])
            if printResults: print(f"Score: {ranking}; Tweet:{tweet}")

        print("done")
        resultsDF = pd.DataFrame(resultsList, columns = ["score", "id", "tweet"])

        return resultsDF


    except Exception as e:
        print(e)
        print("there was an error")




if __name__ == '__main__':
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])

    #ensure query has been passed as command line argument.
    query = ""
    if len(sys.argv) == 1:
        print("usage: make_query.py '<query string>' <results count (optional)>")
        print("(multi-word queries must be surrounded by quotation marks)")
        print("exiting")
        sys.exit()

    query = sys.argv[1]
    resultCount = 50

    #setting result count (if specified by user, otherwise defaults to 50)
    if len(sys.argv) > 2:
        try:
            resultCount = int(sys.argv[2])
        except:
            print("usage: make_query.py '<query string>' <results count (optional)>")
            print("(multi-word queries must be surrounded by quotation marks)")
            print("exiting")
            sys.exit()

    #making query
    results = lucene_query(query, resultCount)
    
    print(results)
    print("exiting")
