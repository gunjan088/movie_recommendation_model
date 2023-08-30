

import pandas as pd
ratings = pd.read_csv(r'D:\Nidhi\MA\MA\SEM-6\BDA\ml-25m\ratings.csv' ,delimiter=',',dtype = {0:int, 1:int, 2:float, 3: int})

from scipy import sparse
TrainSparseData = sparse.csr_matrix((ratings.rating, (ratings.userId, ratings.movieId)))

#looking at distribution of ratings
avgratinguser = ratings.groupby('userId').mean().rating

avgratingmovie = ratings.groupby('movieId').mean().rating

#finding similar users based on knn on user-movie matrix
from sklearn.neighbors import NearestNeighbors
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model_knn.fit(TrainSparseData)

distance, indices = model_knn.kneighbors(TrainSparseData.getrow(1))

model_knn.kneighbors_graph(TrainSparseData.getrow(1)).toarray()

user_movie_list = ratings.groupby('userId')['movieId'].apply(list)

#Getting a subset of movies watched by similar movies
flat_list = []
for i in user_movie_list:
  flat_list.extend(i)
flat_list = pd.Series(flat_list).unique()
flat_list = pd.DataFrame(flat_list)
flat_list.rename(columns = {0:'movieId'}, inplace = True)
distance.max()

#importing csvs to get tags of movies
scores = pd.read_csv(r'D:\Nidhi\MA\MA\SEM-6\BDA\ml-25m\genome-scores.csv' ,delimiter=',')
tags = pd.read_csv(r'D:\Nidhi\MA\MA\SEM-6\BDA\ml-25m\genome-tags.csv' ,delimiter=',')
movies = pd.read_csv(r'D:\Nidhi\MA\MA\SEM-6\BDA\ml-25m\movies.csv' ,delimiter=',')
scores = scores.merge(tags, on = 'tagId')

scores['rank'] = scores.groupby("movieId")["relevance"].rank(method = "first", ascending = False).astype('int64')
scores.head()

#fetching top 100 tags for the subset of movies
tags = scores[scores['rank'] <= 100].groupby(['movieId'])['tag'].apply(lambda x: ','.join(x)).reset_index()
tags['taglist'] = tags['tag'].apply(lambda x: x.split(','))
tags = tags.merge(movies, on = 'movieId')
#tags = tags[tags['movieId'].isin(list(flat_list))]

#testing for target movie
target_movie = 1704
target_tag_list = tags[tags['movieId'] == target_movie].taglist.values[0]

#importing nlp libraries
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stop_words = stopwords.words('english')

from nltk.tokenize import word_tokenize

#function for tokenize document and clean
def word_tokenize_clean(doc):
  
  # split into lower case word tokens
  tokens = word_tokenize(doc.lower())
  
  # remove tokens that are not alphabetic (including punctuation) and not a stop word
  tokens = [word for word in tokens if word.isalpha() and not word in stop_words]
  
  return tokens

#importing doc2vec model for finding similar movies based on tags
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
nltk.download('punkt')
tags_values = tags.tag.values
# create tagged individual document of tags of each movie
mv_tags_doc = [TaggedDocument(words=word_tokenize_clean(D), tags=[str(i)]) for i, D in enumerate(tags_values)]

# instantiate Doc2Vec model

max_epochs = 50
vec_size = 20
alpha = 0.025

model = Doc2Vec(vector_size = vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm=0) # paragraph vector distributed bag-of-words (PV-DBOW)

#build vocabulary  
model.build_vocab(mv_tags_doc)

# train Doc2Vec model
# stochastic (random initialization), so each run will be different unless you specify seed

print('Epoch', end = ': ')
for epoch in range(max_epochs):
  print(epoch, end = ' ')
  model.train(mv_tags_doc,
              total_examples=model.corpus_count,
              epochs=model.epochs)
  # decrease the learning rate
  model.alpha -= 0.0002
  # fix the learning rate, no decay
  model.min_alpha = model.alpha

#finding most similar movies after training the model
sims = model.docvecs.most_similar(positive = [1704], topn = 30)

#printing recommendations
for i, j in sims:
  print(tags[tags.movieId == int(i)]['title'], j)







