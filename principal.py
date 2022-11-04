import streamlit as st

st.title('naive bayes IA iris')
st.write ('criando IA iris')

from sklearn.naive_bayes import GaussianNB 
import numpy as np  #importa dados csv
dados=pd.read_csv('/content/drive/MyDrive/IA claudio/Iris - Iris (1).csv')
st.code #importa dados csv

  
