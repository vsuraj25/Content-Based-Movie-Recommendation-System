import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

def preprocess():
    ## Loading the data
    credit_df = pd.read_csv('data/tmdb_5000_credits.csv')
    movies_df = pd.read_csv('data/tmdb_5000_movies.csv')

    ## Merging both the datasets
    movies_df = movies_df.merge(credit_df, on='title')

    ## Dropping unnecessary columns
    cols = ['genres', 'id', 'keywords',
        'original_title', 'overview',
        'title', 'cast', 'crew']

    movies_df = movies_df[cols]

    # Approach 
    # For this project, the approch is to combine the overview, genre, director, top 3 actors and keywords of the movies and create a single column tag which will help us to get all the required details about the movie easily.

    ## Checking Missing Values
    movies_df.dropna(inplace = True)

    ## Getting genres name from genres dictionary
    movies_df['genres'] = movies_df['genres'].apply(get_name_from_dict)

    ## Getting keywords name from keywords dictionary
    movies_df['keywords'] = movies_df['keywords'].apply(get_name_from_dict)

    ## Getting top 3 actor name from cast dictionary
    movies_df['cast']  = movies_df['cast'].apply(get_top_3_actor_name_from_dict)

    ## Getting director name from crew dictionary
    movies_df['crew']  = movies_df['crew'].apply(get_director_name_from_dict)

    ## Convetring overview into a list for easy concatination
    movies_df['overview'] = movies_df['overview'].apply(lambda x:x.split())

    ## Removing spaces from all the values in the keywords, genres, cast column
    movies_df['genres'] = movies_df['genres'].apply(lambda x: [i.replace(' ', '') for i in x])
    movies_df['keywords'] = movies_df['keywords'].apply(lambda x: [i.replace(' ', '') for i in x])
    movies_df['cast'] = movies_df['cast'].apply(lambda x: [i.replace(' ', '') for i in x])
    movies_df['crew'] = movies_df['crew'].apply(lambda x: [i.replace(' ', '') for i in x])

    ## Concatinating all the columns except id and title into a single column 'tags'
    movies_df['tags'] =  movies_df['overview'] + movies_df['genres'] + movies_df['keywords'] + movies_df['cast'] + movies_df['crew']

    ## New Dataframe
    new_df =  movies_df[['id', 'original_title', 'tags']]

    ## Coverting list into string
    convert_list_to_string(new_df['tags'][0])
    new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))

    ## Lower Casing the tags
    new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
 
    ## Applying Stemming
    new_df['tags'] = new_df['tags'].apply(stem)

    ## Vectorizing the tags
    cv = CountVectorizer(max_features=5000, stop_words='english')
    movie_vectors = cv.fit_transform(new_df['tags']).toarray()

    ## Calculating distance between vectors using cosine similarity
    similarity = cosine_similarity(movie_vectors)

    return new_df, similarity

## For converting a stringed list into list
def get_name_from_dict(list_of_dic):
    name = []
    for l in ast.literal_eval(list_of_dic):
        name.append(l['name'])
    return name

def get_top_3_actor_name_from_dict(list_of_dic):
    name = []
    top_3_counter = 0
    for l in ast.literal_eval(list_of_dic):
        if top_3_counter != 3:
            name.append(l['name'])
            top_3_counter += 1
        else:
            break
    return name

def get_director_name_from_dict(list_of_dic):
    name = []
    for l in ast.literal_eval(list_of_dic):
        if l['job'] == 'Director':
            name.append(l['name'])
            break
    return name

def convert_list_to_string(input_list):
    string = ''
    for i in input_list:
        string += ' '.join(i)
    return string

def stem(text):
    stemmer = PorterStemmer()
    stem_text = []
    for i in text.split():
        stem_text.append(stemmer.stem(i))
    return " ".join(stem_text)