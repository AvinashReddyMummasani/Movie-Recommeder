import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import streamlit as st

embed=pd.read_csv('embeddings.csv').values

movie_data=pd.read_csv("movies.csv")
id=movie_data['id']
movie_name=movie_data['Title']
id=list(id)
movie_name=list(movie_name)

ne=NearestNeighbors(n_neighbors=6,algorithm='brute',metric='cosine')
ne.fit(embed)
st.title("Movie Recommender System")
selected_movie=st.selectbox("Select Movie:",movie_name)

#https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&


if st.button("Recommend"):

    selected_ind=movie_name.index(selected_movie)

    dist,ind=ne.kneighbors(embed[selected_ind].reshape(1, -1))

    for i in ind[0][1:]:
        st.write(movie_name[i])
        st.write(" ")
