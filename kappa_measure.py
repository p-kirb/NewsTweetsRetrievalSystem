import pandas as pd


def loadJudgements():
    judgements = pd.read_csv("data/kappa_test/judgement_vectors.csv")
    return judgements

def calculateFliessKappa(judgements):
    #make 2x75 matrix
    noJudges = len(judgements.columns)
    tweetRelevance = pd.DataFrame()
    tweetRelevance["relevant"] = judgements.agg([sum], axis=1)
    tweetRelevance["irrelevant"] = tweetRelevance["relevant"].apply(lambda x : noJudges-x)

    pj = {}                 #proportion of assignments in relevant/not relevant
    pj[0] = judgements.sum().sum()
    pj[1] = judgements.size - pj[0]
    pj[0] = pj[0] / (judgements.size * noJudges)
    pj[1] = pj[1] / (judgements.size * noJudges)


    #inter-judge agreement of relevance per tweet
    pi = tweetRelevance.apply(lambda row : (row.apply(lambda x : x**2).sum() - noJudges) / (noJudges * (noJudges-1)), axis=1)    #(each element^2 - no of judges) / no of judges*(no of judges -1)

    pBar = pi.sum() / pi.size
    
    pBare = (pj[0] ** 2) + (pj[1] ** 2)

    kappa = (pBar - pBare)/(1-pBare)
    
    return kappa


def _calculateCohensKappa(judgements):
    '''
    calculates Cohens kappa between first 2 columns of input dataframe.
    '''
    #prob. agreement - prob agreement by chance / prob disagreement by chance

    noSubjects = len(judgements.index)

    agreements = judgements.apply(lambda x : int(x.iloc[0] == x.iloc[1]), axis=1)

    probAgreement = sum(agreements)/noSubjects

    rel0 = judgements.iloc[:,0].sum()          #number of documents considered relevant
    rel1 = judgements.iloc[:,1].sum()

    probChanceAgreement = ((rel0/noSubjects)*(rel1/noSubjects))+(((noSubjects-rel0)/noSubjects)*((noSubjects-rel1)/noSubjects)) 

    kappa = (probAgreement - probChanceAgreement) / (1 - probChanceAgreement)
    
    print(probAgreement)
    print(probChanceAgreement)
    print(kappa)

    return kappa

def calculateMeanCohensKappa(judgements):
    kappas = []
    kappas.append(_calculateCohensKappa(judgements[["JudgeA", "JudgeB"]]))
    kappas.append(_calculateCohensKappa(judgements[["JudgeB", "JudgeC"]]))
    kappas.append(_calculateCohensKappa(judgements[["JudgeA", "JudgeC"]]))


    return sum(kappas)/len(kappas)


if __name__ == "__main__":
    judgements = loadJudgements()

    resultsString = "Relevance and agreement statistics of top 15 results from 5 queries.\n\n"

    #% relevance
    perJudgeRelevance = judgements.mean()*100
    resultsString = "".join((resultsString,"Opinion of percentage relevance of results:\n", perJudgeRelevance.to_string(), "\n"))
   
    #avg relevance
    avgRelevance = perJudgeRelevance.mean()
    resultsString = "".join((resultsString,"Average   ", str(avgRelevance), "\n\n"))

    #kappa measures
    fliessKappa = calculateFliessKappa(judgements)
    resultsString = "".join((resultsString, "Fliess' Kappa: ", str(fliessKappa), "\n"))

    meanCohens = calculateMeanCohensKappa(judgements)
    resultsString = "".join((resultsString, "Mean Cohen's Kappa: ", str(meanCohens), "\n"))

    print(resultsString)

    f = open("data/kappa_test/relevance_stats.txt", "w")
    f.write(resultsString)
    f.close()

    