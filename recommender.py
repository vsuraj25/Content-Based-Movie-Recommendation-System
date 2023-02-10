import pandas as pd 
import requests

def fetch_poster(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US'.format(movie_id))
    data = response.json()
    return "https://image.tmdb.org/t/p/original" + data['poster_path']

def recommend(movie, dataframe, similarity_matrix):
    movie_index = dataframe[dataframe['original_title'] == movie].index[0]
    distances =  similarity_matrix[movie_index]
    rec_movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]

    recommended_movies = []
    recomended_movies_poster = []
    for i in rec_movie_list: 
        movie_id = dataframe.iloc[i[0]].id 
        ## Fetching movie poster
        recommended_movies.append(dataframe.iloc[i[0]].original_title)
        recomended_movies_poster.append(fetch_poster(movie_id))
    return recommended_movies, recomended_movies_poster
