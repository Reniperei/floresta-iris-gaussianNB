import streamlit as st

st.title('naive bayes IA iris')
st.write ('criando IA iris')

from sklearn.ensemble import Randonforestclassifier
import pandas as pd #importa dados csv
dados=pd.read_csv('/content/drive/MyDrive/IA claudio/Iris - Iris (1).csv')
