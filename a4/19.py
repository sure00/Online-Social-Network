import matplotlib.pyplot as plt
import os
import pandas as pd
import urllib
import zipfile
import numpy as np
from scipy.stats import pearsonr

def correlation(v1, v2):
    indices=[i for i in range(len(v1)) if v1[i]!=0 and v2[i]!=0]
    print(v1[indices])
    print(v2[indices])
    if len(indices) < 2:
        return 0
    else:
        return pearsonr(v1[indices], v2[indices])[0]

def download_data():
    """
    Download and unzip data.
    :return:
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


download_data()
path = 'ml-latest-small'
ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
movies = pd.read_csv(path + os.path.sep + 'movies.csv')
tags = pd.read_csv(path + os.path.sep + 'tags.csv')
#print(ratings.head(3))
#print(movies.head(3))
#print(tags.head(3))
#print(movies[movies.movieId==3671].iloc[0]['genres'])
user_ids = sorted(set(ratings.userId))
#print(user_ids[:10])
#print(ratings[ratings.movieId==3671])

target_movie_id = 3671
target_movie_vector = np.zeros(len(user_ids)+1)
for index, row in ratings[ratings.movieId==3671].iterrows():
    #print("row is", row)
    #print("row userId is", row.userId)
    #print("row rating is", row.rating)
    target_movie_vector[int(row.userId)] = row.rating
target_movie_vector[0] = 0
#print(target_movie_vector)

ret = correlation(np.array([0, 4, 0, 5, 0, 5, 2]),
            np.array([4, 3, 0, 4, 0, 5, 1]))

print(ret)