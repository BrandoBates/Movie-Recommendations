import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3), encoding='latin-1')

m_cols = ['movie_id', 'title']
movies = pd.read_csv('ml-100k/u.item', sep='|', names=m_cols, usecols=range(2), encoding='latin-1')


## Recommends movies by highest average rating ##

#Create data frame of user_ids, movie_ids, and ratings
popularity_model = pd.DataFrame(data=ratings)

#Group the data frame by movie_id and sort by the top 20 average ratings
top_movies = popularity_model.groupby('movie_id')['rating'].mean().sort_values(ascending=False).head(20)
#print(top_movies.head())


## Recommends movies via Item-based Collaborative Filtering ##

#Merge movies and ratings data frames
ratings = pd.merge(movies, ratings)
#print(ratings.head())

#Create data frame for each user's ratings
all_ratings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
#print(all_ratings.head())

#Create a correlation matrix of movies 
corr_matrix = all_ratings.corr(method='pearson', min_periods=100)
#print(corr_matrix.head())

#Create list of movies and ratings from user 1
#THIS IS WHERE YOU SELECT WHICH USER
my_ratings = all_ratings.loc[10].dropna()
#print(my_ratings)

#Simulate candidates for movie recommendations
simulations = pd.Series()
for i in range(0, len(my_ratings.index)):
    sims = corr_matrix[my_ratings.index[i]].dropna()
    sims = sims.map(lambda x: x * my_ratings[i])
    simulations = simulations.append(sims)

#Sum duplicate movie scores
simulations = simulations.groupby(simulations.index).sum()
simulations.sort_values(inplace = True, ascending = False)

#Remove movies already rated
simulations = simulations.drop(my_ratings.index, errors='ignore')
print(simulations.head(10))
