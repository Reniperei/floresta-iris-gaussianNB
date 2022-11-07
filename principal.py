import streamlit as st
from sklearn.naive_bayes import GaussianNB
dados=pd.read_csv('Iris -Iris.csv')
st.title('naive bayes IA iris')
st.write ('criando IA')

import pandas as pd
classes=dados['Species']
nomesColunas=dados.columns.to_list()
tamanho=len(nomesColunas)
nomesColunas= nomesColunas[1:tamanho-1]
features=dados[nomesColunas]
from sklearn.model_selection import train_test_splitfeatures_treino,features_teste,classes_treino,classes_teste= train_test_split(features,classes,test_size=0,26,randon_state=3)
model=GaussianNB()

floresta = RandomForestClassifier(n_estimators=90) 

floresta.fit(features_treino,classes_treino)
predicoes = floresta.predict(features_teste)
from sklearn import metrics 
print(metrics.classification_report(classes_teste,predicoes))


                              



 

    
  
   

    


  


  
