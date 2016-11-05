# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.
    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.
    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    e = pd.Series([tokenize_string(gen) for gen in movies['genres'].tolist() if gen])
    movies = movies.assign(tokens=e.values)

    return movies


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i
    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    #print("featurize movies is\n",movies['genres'].tolist())
    uniqueTerm = Counter()
    terms=[]

    for gen in movies['genres'].tolist():
        #print("gen is ", )
        if gen  == '(no genres listed)' :
            #print("no genres listed found")
            terms +=[Counter(['no','genres','listed'])]
            #print("terms %s , splite to %s" %(gen, Counter(['no','genres','listed'])))
        else:
            terms += [(Counter(gen.lower().split('|')))]
            #print("terms %s, splite to %s " %(gen, Counter((gen.lower()).split('|'))))
    #print("each movie term status is",terms)

    #number of unique documents containing term i
    for feat in terms:
        uniqueTerm.update(list(feat.keys()))

    vocab = dict(zip(list(sorted(uniqueTerm.keys(), key=lambda s: s.lower())), list(range(len(uniqueTerm)))))
    #print("vocab is %s, its len is %d" %(sorted(vocab.items()), len(vocab)))

    #print("count is",uniqueTerm)

    N = movies.shape[0]

    featuresValue = []
    for feat in terms:
        col = []
        row = []
        tfidf = []
        for term in feat.keys():
            row.append(0)
            col.append(vocab[term])
            tfidf.append(1.0*feat[term]/max(feat.values())* math.log10(N/uniqueTerm[term]))
        #print("row is %s, col is %s, tfidf is %s" %(row, col, tfidf))
        X=csr_matrix((tfidf, (row, col)), shape=(1, len(vocab)))
        #print("x is",X.toarray())
        featuresValue.append(X)


    e = pd.Series(featuresValue)
    movies = movies.assign(features=e.values)

    return movies, vocab


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    #print("a",a.toarray())
    #print("b",b.toarray())

    normA = np.sqrt((a.toarray() ** 2).sum())
    normB = np.sqrt((b.toarray() ** 2).sum())

    #print("normA", normA)
    #print("normB", normB)

    res =  (1.0* np.dot(a.toarray()[0], b.toarray()[0]) / (normA * normB))

    return res


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.
    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.
    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    #print("ratings_test is", ratings_test)
    #print("ratings_train is", ratings_train)

    res=[]
    for index, rowsInTest in ratings_test.iterrows():
        tmp = []
        rate = []
        #print("index is %d, rowsInTest is %s" %(index, rowsInTest))
        #print("ratings_train\n", ratings_train)
        ratingTestuserID = ratings_train['userId'] == rowsInTest['userId']
        #print("userId is %d\n, resutlt is %s \n" %(rowsInTest['userId'], ratings_train[ratingTestuserID]))
        for index, rowsinTrain in ratings_train[ratingTestuserID].iterrows():
            TrainMoveId = movies['movieId']== rowsinTrain['movieId']
            TestMoveId = movies['movieId']== rowsInTest['movieId']
            #print("movies-TrainMove %s\n, movies-TestMove %s\n" %(movies[TrainMoveId],movies[TestMoveId]))
            #print("TrainMoveID is %s \n, testMoveId is %s \n" %(movies[TrainMoveId]['features'].tolist()[0], movies[TestMoveId]['features'].tolist()[0]))
            #print("a is %s\n" %movies[TrainMoveId]['features'].tolist())
            #print("b is %s\n" %movies[TestMoveId]['features'].tolist())
            cosSim = cosine_sim(movies[TrainMoveId]['features'].tolist()[0], movies[TestMoveId]['features'].tolist()[0])

            if cosSim > 0:
                tmp.append(cosSim)
            else:
                tmp.append(0)
            #print("rate is ",rowsinTrain['rating'])
            rate.append(rowsinTrain['rating'])
        if sum(tmp)==0:
            res.append(np.mean(rate))
        else:
            res.append((sum(np.array(tmp) * np.array(rate))/sum(tmp)))
    return res


def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
