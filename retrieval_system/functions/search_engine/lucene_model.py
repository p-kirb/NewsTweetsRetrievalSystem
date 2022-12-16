import lucene
import pandas as pd
from java.nio.file import Paths
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.search import IndexSearcher


def lucene_query(queryString, noOfResults, components_path, verbose=False):
    '''
    Makes a query to the Lucene index based on given query string.
    
            Parameters:
                queryString (string): Query used to retrieve documents.
                noOfResults (int): Upper bound of results that should be returned by query.
                printResults (boolean): function prints results if true.

                Returns:
                    resultsDF (pandas.DataFrame): Dataframe with columns "score", "id", and "tweet" containing the results.
    '''
    if verbose: print(f"Retrieving top {noOfResults} results for query: '{queryString}'")

    dirPath = Paths.get(components_path)
    directory = FSDirectory.open(dirPath)
    analyzer = EnglishAnalyzer()                   #removal of stopwords in query is unneccesary as they are not indexed.
    
    resultsList = []

    try:
        searcher = IndexSearcher(DirectoryReader.open(directory))
        query = QueryParser("tweet", analyzer).parse(queryString)
        topDocs = searcher.search(query, noOfResults)
        if verbose: print(f"Got {topDocs.totalHits.value} results:")
        for result in topDocs.scoreDocs:
            ranking = result.score
            tweet = searcher.doc(result.doc).get("tweet")
            id = searcher.doc(result.doc).get("id")
            resultsList.append([ranking, id, tweet])
            if verbose: print(f"Score: {ranking}; Tweet:{tweet}")

        # print("done")
        resultsDF = pd.DataFrame(resultsList, columns = ["score", "id", "tweet"])

        return resultsDF


    except Exception as e:
        print(e)
        print("there was an error")