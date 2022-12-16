import sys
import lucene
import pandas as pd
from java.nio.file import Paths
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.search import IndexSearcher
from functions.search_engine.lucene_model import lucene_query


LUCENE_COMPONENTS_PATH = "./data/04_model_components/lucene_components"


if __name__ == '__main__':
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])

    # Ensure query has been passed as command line argument.
    query = ""
    if len(sys.argv) == 1:
        print("usage: make_query.py '<query string>' <results count (optional)>")
        print("(multi-word queries must be surrounded by quotation marks)")
        print("exiting")
        sys.exit()

    query = sys.argv[1]
    resultCount = 50

    # Setting result count (if specified by user, otherwise defaults to 50)
    if len(sys.argv) > 2:
        try:
            resultCount = int(sys.argv[2])
        except:
            print("usage: make_query.py '<query string>' <results count (optional)>")
            print("(multi-word queries must be surrounded by quotation marks)")
            print("exiting")
            sys.exit()

    # Making query
    results = lucene_query(query, resultCount, LUCENE_COMPONENTS_PATH, verbose=True)
    
    print(results)
    print("exiting")