import streamlit as st
import pandas as pd
dados= pd.read_csv('Iris.csv')
st.title('naive bayes IA iris')
st.write ('criando IA')


classes=dados['Species']
nomesColunas=dados.columns.to_list()
tamanho=len(nomesColunas)
nomesColunas= nomesColunas[1:tamanho-1]
features=dados[nomesColunas]
from sklearn.model_selection import train_test_split
features_treino,features_teste,classes_treino,classes_teste= train_test_split(features,classes,test_size=0.3,randon_state=4)

model=GaussianNB()


model.fit(features_treino,classes_treino)
predicoes=model.predict(features_teste)

st.title('naive bayes IA iris')
SepalLengthCm=st.number_input('digite comprimento do caule')
SepalWidthCm=st.number_input('digite largura do caule')
PetalLengthCm=st.number_input('digite comprimento da petala')
PetalWidthCm=st.number_input('digite largura da petala')
if st.button('aplicar'):
 resultado=model.predict([[SepalLengthCm,SepalWidthCm,PetalLenthCm,PetalWidthCm]])
                             
 if resultado==('Iris-setosa'):
  st.write('setosa')
  st.image('iris_setosa.jpg')
                             
 if resultado==('Iris-versicolor'):
  st.write('versicolor')
  st.image('iris_versicolor.jpg')
                             
 if resultado==('Iris-virginica'):
  st.write('virginica')
  st.image('iris_virginica.jpg')
                              

#resultado de previsão

teste = np.array([[-4,0],[0,0],[-2,0],[2,7]])
#predicted = model.predict([[1,2][0,0]]) #fazer a previsão em cima desses 2 numeros
predicted = model.predict(teste) #fazer a previsão em cima desse array teste
print(predicted)

#comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, predicted)*100)



                              



 

    
  
   

    


  


  
