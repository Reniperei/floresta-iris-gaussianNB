import streamlit as st

st.title('naive bayes IA iris')
st.write ('criando IA iris')
st.number_input
st.text_input
st.file_uploader
st.image
st.columns
st.tabs
st.container
st.dataframe
st.metric
st.experimental_show
st.pyplot
st.code

from sklearn.ensemble import RandomForestClassifier
import pandas as pd 

                importando os dados do csv                          

dados = pd.read_csv('/content/drive/MyDrive/IA claudio/Iris - Iris (1).csv')
           separar as classes das features                         
classes = dados['Species']
nomesColunas = dados.columns.to_list()
tamanho = len(nomesColunas)#quantos nomes tem
nomesColunas = nomesColunas[1:tamanho-1]
features = dados[nomesColunas]#monta o features
           quebrar os dados em teste e treino                      
from sklearn.model_selection import train_test_split

features_treino,features_teste,classes_treino,classes_teste = train_test_split(features,
                                                                               classes,
                                                                               test_size=0.4,
                                                                               random_state=2)

floresta = RandomForestClassifier(n_estimators=1000) 

floresta.fit(features_treino,classes_treino)

predicoes = floresta.predict(features_teste)
from sklearn import metrics

print(metrics.classification_report(classes_teste,predicoes))


from sklearn.naive_bayes import GaussianNB 
import numpy as np  #importa dados csv
dados=pd.read_csv('/content/drive/MyDrive/naive bayes2.ipynb')
#importa a biblioteca do modelo Naive Bayes Gaussiano
#import pandas as pd #manipulacao de dados
from sklearn.naive_bayes import  GaussianNB
import numpy as np


x = np.array([[1,2],[1,2],[1,2],[-2,0],[2,3],[-4,0],[-1,1],[1,2],[-2,2],[2,7],[-4,1],[0,0]])
y = np.array([15, 15, 15, 3, 4, 10, 3, 15, 3, 4, 4,7 ])


# splittin X and y into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

#cria um classificador Gaussiano
model = GaussianNB()

#treina o modelo usando os dandos de treino
model.fit(x_train, y_train)

#resultado de previsão

teste = np.array([[-4,0],[0,0],[-2,0],[2,7]])
#predicted = model.predict([[1,2][0,0]]) #fazer a previsão em cima desses 2 numeros
predicted = model.predict(teste) #fazer a previsão em cima desse array teste
print(predicted)

#comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, predicted)*100)

  


  
