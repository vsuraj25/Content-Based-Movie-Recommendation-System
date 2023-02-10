from recommender import recommend
from preprocessor import preprocess
import streamlit as st
import pandas as pd
import pickle
import _pickle as cPickle
import bz2

movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
data = bz2.BZ2File('similarity.pbz2', 'rb')
similarity_matrix = cPickle.load(data)

movies = pd.DataFrame(movies_dict)
movies_list = movies['original_title'].values 

st.title('Movie Recommendation System')

selected_movie_name = st.selectbox('Choose a movie',movies_list)

if st.button('Recommend'):
    recommendations, movie_posters = recommend(selected_movie_name, movies, similarity_matrix)
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.text(recommendations[0])
        st.image(movie_posters[0])

    with col2:
        st.text(recommendations[1])
        st.image(movie_posters[1])

    with col3:
        st.text(recommendations[2])
        st.image(movie_posters[2])

    with col4:
        st.text(recommendations[3])
        st.image(movie_posters[3])

    with col5:
        st.text(recommendations[4])
        st.image(movie_posters[4])



     