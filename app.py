import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from streamlit import components

st.title('Book Recommendation System')

final = pd.read_csv('final.csv')
pd.set_option('display.width', 1000)


book_pivot = final.pivot_table(columns='user_id',index='title',values='number_of_ratings',fill_value=0)

book_sparse = csr_matrix(book_pivot)

model = NearestNeighbors(algorithm = 'brute')
model.fit(book_sparse)

Book = st.text_input('Enter the Book name!')
if st.button('Recommend'):
    def book_recomend(book):
        books = []  
        book_id = np.where(book_pivot.index==book)[0][0]
        distances, suggestions = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6)
        for i in range(len(suggestions)):
            books.append(book_pivot.index[suggestions[i]])
        return books

    library = book_recomend(Book)

    for item in library:
        st.write('The Recommendations are as follows:', item[1:6])
