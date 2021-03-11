import math
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import time
from collections import OrderedDict
import re
# pip3 install fuzzywuzzy
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
plt.style.use('ggplot')

start_time = time.time()
#Loading movielens dataset(movies, links, tags, ratings)
movies_ml = pd.read_csv("C://users/rikky/Documents/datasets/ml-latest/ml-latest/movies.csv",  usecols = ['movieId', 'title'],
    dtype={'movieId': 'int32', 'title': 'str'})
ratings_ml = pd.read_csv("C://users/rikky/Documents/datasets/ml-latest/ml-latest/ratings.csv",
    usecols=['userId', 'movieId', 'rating'],
    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})


#Prepare data for K Nearest Neighbors
mv_list = movies_ml['title'].tolist()
userlist = ratings_ml['userId'].unique().tolist()

# get count of each rating value
ratings_count_tmp = pd.DataFrame(ratings_ml.groupby('rating').size(), columns=['count'])

# there no count for 0 rating, as no user has given it.
#populate the 0 rating value
total_count = len(userlist) * len(mv_list)
rating_zero_count = total_count - ratings_ml.shape[0]
# append counts of zero rating to df_ratings_cnt
ratings_count = ratings_count_tmp.append(
    pd.DataFrame({'count': rating_zero_count}, index=[0.0]),
    verify_integrity=True,
).sort_index()

# add log count
ratings_count['log_count'] = np.log(ratings_count['count'])

# movies rating frequency
movies_count = pd.DataFrame(ratings_ml.groupby('movieId').size(), columns=['count'])
# print(movies_count.head())

# filter data
popularity_threshold = 50
popular_movies = list(set(movies_count.query('count >= @popularity_threshold').index))
ratings_drop_movies_ml = ratings_ml[ratings_ml.movieId.isin(popular_movies)]

# get number of ratings given by each user
users_count = pd.DataFrame(ratings_drop_movies_ml.groupby('userId').size(), columns=['count'])
users_count['count'].quantile(np.arange(1, 0.5, -0.05))

# filter data
ratings_threshold = 50
active_users = list(set(users_count.query('count >= @ratings_threshold').index))
ratings_drop_users = ratings_drop_movies_ml[ratings_drop_movies_ml.userId.isin(active_users)]
# print('shape of original ratings data: ', ratings_ml.shape)
# print('shape of ratings data after dropping both unpopular movies and inactive users: ', ratings_drop_users.shape)

#Reshaping the Data
movie_user_matrix = ratings_drop_users.pivot(index='movieId', columns='userId', values='rating').fillna(0)
# create mapper from movie title to index
movie_to_index = {
    movie: i for i, movie in
    enumerate(list(movies_ml.set_index('movieId').loc[movie_user_matrix.index].title))
}
# transform matrix to scipy sparse matrix
movie_user_matrix_sparse = csr_matrix(movie_user_matrix.values)


# Define model
knnModel = NearestNeighbors(metric='euclidean', algorithm='brute', n_neighbors=20, n_jobs=-1)

def fuzzymatching(mapper, fav_movie, verbose=True):
    """
    return the index of the closest match via fuzzy ratio.
    return None if no match found.
    """
    match_tuple = []
    # getting the match from movie_to_index we populated
    for title, idx in mapper.items():
        ratio = fuzz.ratio(title.lower(), fav_movie.lower())
        if ratio >= 60:
            match_tuple.append((title, idx, ratio))
    # sorting the matched entries
    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
    if not match_tuple:
        print('Oops! No match is found')
        return
    if verbose:
        print("\n");
        # print('Found possible matches in our Dictionary: {0}\n'.format([x[0] for x in match_tuple]))
    return match_tuple[0][1]

def make_recommendation(knnModel, train_data, mapper_for_dict, movie_for_which_neighbors_to_find, no_of_recommendations):
    """
    return top n similar movie recommendations based on user's input movie
    """
    # fit the model
    knnModel.fit(train_data)
    # get input movie index through fuzzy matching technique
    print('User input movie:', movie_for_which_neighbors_to_find)
    index = fuzzymatching(mapper_for_dict, movie_for_which_neighbors_to_find, verbose=True)
    print(index)
    # inference
    print('Finding the most similar movies through KNN')
    print('......\n')
    distances, indices = knnModel.kneighbors(train_data[index], n_neighbors=no_of_recommendations+1)
    # get list of raw indexes of recommendations
    raw_recommends = \
        sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[1::1]
    # find the reverse mapper and display the results
    reverse_mapper = {v: k for k, v in mapper_for_dict.items()}
    #print(reverse_mapper)
    print('Recommendations similar to {}:'.format(movie_for_which_neighbors_to_find))
    for i, (ind, dist) in enumerate(raw_recommends):
        print('{0}: {1}, with distance of {2}'.format(i+1, reverse_mapper[ind], dist))


#K-Means Code
def getKeysByValue(dictOfElements, valueToFind):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for v in valueToFind:
        for item  in listOfItems:
            if item[1] == int(v):
                listOfKeys.append(item[0])
    return  listOfKeys

def get_avg_rating(row):
    val = row.values
    val = val[1:131]
    count = 0
    sum = 0.0
    for i in val:
        if (i != 0.0):
            count += 1
            sum += i

    avg = sum / count
    return avg

kmeans = KMeans(n_clusters=6, init='k-means++', n_jobs=-1)

#Prepare data for K-Means
popular_movies = list(set(movies_count.query('count >= @popularity_threshold').index))
ratings_drop_movies_ml_kmeans = ratings_ml[ratings_ml.movieId.isin(popular_movies)]
# print(len(ratings_drop_movies_ml_kmeans["movieId"].unique().tolist()))

r_threshold=2750
active_users = list(set(users_count.query('count >= @r_threshold').index))
ratings_drop_users_kmeans = ratings_drop_movies_ml_kmeans[ratings_drop_movies_ml_kmeans.userId.isin(active_users)]
# print('shape of original ratings data: ', ratings_ml.shape)
# print('shape of ratings data after dropping both unpopular movies and inactive users: ', ratings_drop_users_kmeans.shape)

#Create pivot
movie_user_matrix_kmeans = ratings_drop_users_kmeans.pivot(index='movieId', columns='userId', values='rating').fillna(0)
#Populate a dict with movie name and ID
movie_to_index_kmeans = {
    movie: i for i, movie in
    enumerate(list(movies_ml.set_index('movieId').loc[movie_user_matrix_kmeans.index].title))
}
#Sparse Matrix
movie_user_matrix_sparse_kmeans = csr_matrix(movie_user_matrix_kmeans.values)
# print(movie_user_matrix_kmeans.shape)

start_time = time.time()
predictions = kmeans.fit_predict(movie_user_matrix_sparse_kmeans)
# print("--- %s seconds ---" % (time.time() - start_time))
clustered = pd.concat([movie_user_matrix_kmeans.reset_index(), pd.DataFrame({'group':predictions})], axis=1)
# print(clustered)

def recommend_kmeans(movie_input):
    ind= fuzzymatching(movie_to_index, movie_input, verbose=True)
    cluster = clustered[clustered.movieId == ind]
    #get avg ratings of the movie passed
    cl = cluster.iloc[0, 1:131]
    val = cl.values
    count = 0
    sum = 0.0
    for i in val:
        if (i != 0.0):
            count += 1
            sum += i

    movie_avg = sum / count

    clust = clustered[clustered["group"]==int(cluster["group"])]
    #get all the movies in the cluster of the movie
    movieIdList = OrderedDict()
    for i,e in clust.iterrows():
        a = get_avg_rating(e)
        if a >= movie_avg:
            movieIdList[e["movieId"]] = a

    movieIdList = OrderedDict(sorted(movieIdList.items(), key=lambda x: x[1]))
    n_items = list(movieIdList.keys())[:15]

    l = getKeysByValue(movie_to_index, n_items)
    print("Recommendations of similar movies through K-means:")
    print('Recommendations similar to {}:'.format(movie_input))
    print(l[1:12])



#Code for preparing data and calculate Highly rated movies based on genre
#Loading kaggle movies dataset
df_kagglemv = pd.read_csv("C://users/rikky/Documents/datasets/kaggledataset/movies_metadata.csv", usecols = ['id','imdb_id','overview','title','popularity','tagline','vote_average','vote_count'],low_memory=False)

#Loading movielens dataset(movies, links, tags, ratings)
movies_genral_ml = pd.read_csv("C://users/rikky/Documents/datasets/ml-latest/ml-latest/movies.csv")
links_ml = pd.read_csv("C://users/rikky/Documents/datasets/ml-latest/ml-latest/links.csv",  usecols = ['movieId','imdbId'])
ratings_general_ml = pd.read_csv("C://users/rikky/Documents/datasets/ml-latest/ml-latest/ratings.csv",low_memory=False)

#Split the genres to individual columns
#Replace categorical value of genre to numerical values
len_df= len(movies_genral_ml['genres'])
#Splitting the names
df1=movies_genral_ml['genres'].str.split('|', expand=True)
#Putting all the things in a list
my_list = []
for j in range(0,df1.shape[1]):
    for i in range(0,len_df):
        my_list.append(str(df1[j][i]))
# print(my_list)
#Generating unique list
unique_list = np.unique(my_list)
unique_list=unique_list.tolist()
# print(unique_list)
#Remove None
unique_list.remove('None')
# Create separate column
for i in range(0, len(unique_list)):
    a = unique_list[i]
    movies_genral_ml[a] = movies_genral_ml.apply(lambda _: '', axis=1)
df = movies_genral_ml
# df.head(5)
#drop title and year
df.drop(['title'],axis=1,inplace=True)
# Putting the dummy variable
for i in range(2,df.shape[1]):
    for j in range(df.shape[0]):
        a =str(df.columns[i])
        if bool(re.search(a,df.loc[j,'genres'])):
            df.iat[j,i]=1
        else:
            df.iat[j,i]=0

#Drop the genres column
df.drop('genres',axis=1,inplace=True)

#Drop the (no genres listed) column
df.drop('(no genres listed)',axis=1,inplace=True)

#Merge Datasets
#Merge movies.csv, links.csv of movielens datasets
df_merge = movies_genral_ml.merge(links_ml)
#Rename imdbid as per kaggle dataset
df_merge.rename(columns={'imdbId':'imdb_id'}, inplace=True)

# movies_ml.head(5)
#As we need it to merge datasets
#convert object data type to string
df_kagglemv['imdb_id'] = df_kagglemv['imdb_id'].convert_dtypes()
#eliminating the prefix 'tt' and leading zeros present in imdb_id column in kaggle dataset
df_kagglemv['imdb_id'] = df_kagglemv['imdb_id'].str[2:].str.lstrip('0')
#converting string to int64
df_kagglemv['imdb_id'] = pd.to_numeric(df_kagglemv['imdb_id'], errors='coerce').fillna(0).astype(np.int64)

#Merging Movielens and Kaggle datasets
merged_df = pd.merge(df_merge, df_kagglemv, on='imdb_id')
df_beforemergingtag = pd.DataFrame()
df_beforemergingtag = merged_df.copy(deep=True)

vote_counts = df_beforemergingtag[df_beforemergingtag['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = df_beforemergingtag[df_beforemergingtag['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()

m = vote_counts.quantile(0.95)

fit_qualification = df_beforemergingtag[(df_beforemergingtag['vote_count'] >= m) & (df_beforemergingtag['vote_count'].notnull()) & (df_beforemergingtag['vote_average'].notnull())][['title','Action','Adventure','Comedy','Thriller','vote_average','vote_count','popularity']]
fit_qualification['vote_count'] = fit_qualification['vote_count'].astype('int')
fit_qualification['vote_average'] = fit_qualification['vote_average'].astype('int')

def weighted_rating(x):
    count = x['vote_count']
    avg = x['vote_average']
    return (count/(count+m) * avg) + (m/(m+count) * C)

fit_qualification['wr'] = fit_qualification.apply(weighted_rating, axis=1)
qualified = fit_qualification.sort_values('wr', ascending=False).head(250)

def findMoviesBasedOnGenre(df,genre):
    count=1
    ls = []
    for i,row in df.iterrows():
        if row[genre]==1:
            print(row['title'])
            ls.append(row)
            count+=1
        if count>=10:
            break;
    return ls

q_df = pd.DataFrame(qualified,columns=['title','Action','Adventure','Comedy','Thriller','Romance','Children','vote_average','vote_count','wr'])



#Test Data
#Getting users whoc have rated less than 200 ratings, which will be different than the trained data
ratings_test_ml = pd.read_csv("C://users/rikky/Documents/ratings_below_200_df.csv",
    usecols=['userId', 'movieId', 'rating'],
    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
#First 1000 column has 26 unique users from 1 to 25 and user 100
#user 100 is new user and not watched/rated any movie
test_data = ratings_test_ml.iloc[1:1000,]

test_movieid = test_data["movieId"].unique()
train_movieId = movie_to_index.values()
not_present = []
for i in test_movieid:
    if i not in train_movieId:
        not_present.append(i)
for i in not_present:
    test_data = test_data.drop(test_data.index[test_data.movieId == i])

#new_user
new_user = test_data.loc[test_data['userId']==100]

#Normal user
normal_user = test_data.loc[test_data['userId']==1]

users = [new_user,normal_user]
# for user in users:
user = new_user
ratings_mean_count = pd.DataFrame(ratings_ml.groupby("movieId")['rating'].mean())
ratings_mean_count['rating_counts'] = pd.DataFrame(ratings_ml.groupby("movieId")['rating'].count())


def getUserTopRatedMovie(df):
    likedmovies = OrderedDict()
    mvlist = df["movieId"]
    for i in mvlist:
        user_rating = float(user.loc[user['movieId'] == i, 'rating'])
        movie_avg_rating = ratings_mean_count.loc[i]['rating']
        if user_rating >= movie_avg_rating:
            likedmovies[i] = user_rating
    return likedmovies

userDict = getUserTopRatedMovie(user)
userDict = sorted(userDict.items(), key=lambda x: x[1],reverse=True)

import math
moviecount = math.ceil(len(userDict)*.25)

if moviecount ==0:
    print("Highly recommended movies for Action genre:")
    findMoviesBasedOnGenre(q_df, 'Action')
    print("\n")
    print("Highly recommended movies for Comedy genre:")
    findMoviesBasedOnGenre(q_df, 'Comedy')
    print("\n")
    print("Highly recommended movies for Thriller genre:")
    findMoviesBasedOnGenre(q_df, 'Thriller')

if moviecount>=1:
    user_favorite = getKeysByValue(movie_to_index, [userDict[0][0]])[0]
    make_recommendation(
        knnModel=knnModel,
        train_data=movie_user_matrix_sparse,
        mapper_for_dict=movie_to_index,
        movie_for_which_neighbors_to_find=user_favorite,
        no_of_recommendations=10)
if moviecount>1:
    for i in range(1,moviecount):
        user_favorite = getKeysByValue(movie_to_index, [userDict[i][0]])[0]
        recommend_kmeans(user_favorite)

print("--- %s seconds ---" % (time.time() - start_time))



# frame = pd.DataFrame(movie_user_matrix_sparse_kmeans)
# frame['cluster'] = predictions
# frame['cluster'].value_counts()


#Code to select k clusters
#import matplotlib.pyplot as plt
##%matplotlib inline
# fitting multiple k-means algorithms and storing the values in an empty list
# SSE = []
# for cluster in range(2,10):
#     kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init='k-means++')
#     kmeans.fit(movie_user_matrix_sparse_kmeans)
#     SSE.append(kmeans.inertia_)
#
# # converting the results into a dataframe and plotting them
# frame = pd.DataFrame({'Cluster':range(2,10), 'SSE':SSE})
#
# plt.figure(figsize=(12,6))
# plt.plot(frame['Cluster'], frame['SSE'], marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('Inertia')