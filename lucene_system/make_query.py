import sys, os, lucene
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader, IndexWriter, IndexWriterConfig
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.search import IndexSearcher, TopDocs, ScoreDoc




if __name__ == '__main__':
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])

    query = ""
    if len(sys.argv) == 1:
        print("usage: make_query.py '<query string>'")
        print("exiting")
        sys.exit()
    query = sys.argv[1]

    dirPath = Paths.get("./lucene_index")
    directory = FSDirectory.open(dirPath)
    analyzer = StandardAnalyzer()                   
    
    try:
        searcher = IndexSearcher(DirectoryReader.open(directory))
        query = QueryParser("content", analyzer).parse(query)
        topDocs = searcher.search(query, 50)
        for result in topDocs.scoreDocs:
            ranking = result.score
            tweet = searcher.doc(result.doc).get("content")
            #print(result)
            print(f"Score: {ranking}; Tweet:{tweet}")

        print("done")

    except Exception as e:
        print(e)
        print("there was an error")

    print("exiting")
