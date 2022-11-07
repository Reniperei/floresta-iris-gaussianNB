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

treina o modelo usando os dandos de treino
model.fit(x_train, y_train)

#resultado de previsão

teste = np.array([[-4,0],[0,0],[-2,0],[2,7]])
#predicted = model.predict([[1,2][0,0]]) #fazer a previsão em cima desses 2 numeros
predicted = model.predict(teste) #fazer a previsão em cima desse array teste
print(predicted)

#comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, predicted)*100)



                              



 

    
  
   

    


  


  
