# -*- coding: utf-8 -*-

import pandas as pd
ratings = pd.read_csv(r'D:\Nidhi\MA\MA\SEM-6\BDA\ml-25m\ratings.csv' ,delimiter=',',dtype = {0:int, 1:int, 2:float, 3: int})

#adding prefixes to distinguish between user and movie id
ratings['user'] = 'user_' + ratings['userId'].astype(str)
ratings['movie'] = 'movie_' + ratings['movieId'].astype(str)

#getting unique movies and users for creating nodes in graph
movies = ratings['movie'].unique().tolist()
users = ratings['user'].unique().tolist()

#reading csvs
scores = pd.read_csv(r'D:\Nidhi\MA\MA\SEM-6\BDA\ml-25m\genome-scores.csv' ,delimiter=',')
tags = pd.read_csv(r'D:\Nidhi\MA\MA\SEM-6\BDA\ml-25m\genome-tags.csv' ,delimiter=',')
moviesdf = pd.read_csv(r'D:\Nidhi\MA\MA\SEM-6\BDA\ml-25m\movies.csv' ,delimiter=',')

#some movies have not been rated -- removing them
moviesdf = moviesdf[moviesdf.movieId.isin(ratings.movieId.unique().tolist())]
scores = scores.merge(tags, on = 'tagId')

#gtting unique tags
tag_list = scores['tagId'].unique().tolist()
len(tag_list)

#getting unique genres
moviesdf['genre_list'] = moviesdf['genres'].apply(lambda x: x.split('|'))
total_genres_list = []
for i in moviesdf['genre_list']:
  total_genres_list.extend(i)


#creating graph
import networkx as nx
movie_graph = nx.Graph()

#adding nodes in  graph
movie_graph.add_nodes_from(users, type='user')
movie_graph.add_nodes_from(movies, type='movie')
movie_graph.add_nodes_from(total_genres_list , type='genre')
movie_graph.add_nodes_from(tags, type='tag')

#generating movie_genre edges
movie_genre_edges = moviesdf[['movieId', 'genre_list']]
for _, row in movie_genre_edges.iterrows():
    movie_id = 'movie_' + str(row['movieId'])
    genres = row['genre_list']
    for genre in genres:
        movie_graph.add_edge(movie_id, genre, type='belongs to')

#creating user_movie edges with weights as ratings
user_movie_ratings = ratings[['user', 'movie', 'rating']][20000000:].dropna()
for _, row in user_movie_ratings.iterrows():
    user_id = row['user']
    movie_id = row['movie']
    rating = row['rating']
    movie_graph.add_edge(user_id, movie_id, type='rated', rating=rating)
    
import numpy as np

# define the meta-paths to use
meta_paths = ['user','movie', 'genre', 'movie']
#, ['user','movie', 'tag', 'movie']

# define the number of random walks to generate
num_walks = 10

# define the length of each random walk
walk_length = 100

# define the starting node type for the random walks
start_node_type = 'user'

target_user = 'user_130048'

# define a function to generate a random walk based on a given meta-path
def generate_random_walk(meta_path):
    walk = []
    movies = []
    current_node =target_user
    target_movie = ' '
    #current_node_type = start_node_type
    for i in range(walk_length):
        #print('current_node:', current_node)
        #getting next neighbors of the current node
        neighbors = list(movie_graph.neighbors(current_node))
        #print(neighbors)
        neighbor_nodes = [n for n in neighbors if movie_graph.nodes[n]['type'] == meta_path[(i+1) % len(meta_path)]]
        #print("neighbour:", neighbor_nodes)
        if len(neighbor_nodes) == 0:
            break
        #next_node = random.choice(neighbor_nodes)
        #choosing next nodes randomly based on probability if it is a weighted edge like user-movie edge
        if (movie_graph.get_edge_data(current_node,neighbor_nodes[0])['type']=='rated'):
            edge_weight_sum = sum((2.71)**movie_graph[current_node][neighbor]['rating'] for neighbor in neighbor_nodes)
            edge_weights = [(2.71**movie_graph[current_node][neighbor]['rating']) / edge_weight_sum for neighbor in neighbor_nodes]
            next_node = np.random.choice(neighbor_nodes, p=edge_weights)
        else:
            #choosing next nodes randomly if it is an unweighted edge
            next_node = np.random.choice(neighbor_nodes)
        #print("next_node = ", next_node, 'type =', movie_graph.nodes[next_node]['type'])
        current_node = next_node
        walk.append(next_node)
        #print(walk)
        if (movie_graph.nodes[next_node]['type']=='movie'):
            movies.append(next_node)
        #print(movies)
        if (movie_graph.nodes[next_node]['type']=='movie'):
            target_movie = next_node
        #current_node_type = movie_graph.nodes[next_node]['type']
    return target_movie

# generate random walks for each meta-path
random_walks = []
# extract the recommended movies from the random walks
recommended_movies = set()

#unning random walk of length 100 for 10 times
for i in range(num_walks):
    result = generate_random_walk(meta_paths)
    print(result)
    recommended_movies.add(result)

#fetching details of recommended movies
moviesdf['movie'] = 'movie_' + moviesdf['movieId'].astype(str)
recommendations = moviesdf[moviesdf.movie.isin(recommended_movies)][['movie','title','genre_list']]
# print the recommended movies
print(recommendations)



    

