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
from sklearn.model_selection import train_test_split
features_treino,features_teste,classes_treino,classes_teste= train_test_split(features,classes,test_size=0,26,randon_state=3)
model=GaussianNB()

floresta = RandomForestClassifier(n_estimators=90) 

floresta.fit(features_treino,classes_treino)
predicoes = floresta.predict(features_teste)
from sklearn import metrics 
print(metrics.classification_report(classes_teste,predicoes))

                                                                              
  
# comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)

# Import function to fetch dataset
from sklearn.datasets import load_iris

# Import Multinomial Naive-Bayes classifier
from sklearn.naive_bayes import GaussianNB

# Import train_test_split
from sklearn.model_selection import train_test_split

# Import sklearn metrics for analysis
from sklearn.metrics import classification_report, confusion_matrix

def iris_example():

 
 from sklearn.datasets import load_iris
 # Load the data
 iris_df = load_iris(as_frame=True)
 iris_df.target.replace({0: 'setosa', 1: 'versicolor', 2: 'virginica'}, inplace=True)
 # Split the data
 iris_features_train, iris_features_test, iris_species_train, iris_species_test = train_test_split(iris_df.data, iris_df.target, train_size=0.6, shuffle=True, random_state=42)
   # Instantiate a classifier
 classifier = GaussianNB()
                
  # Train the classifier
 classifier.fit(iris_features_train, iris_species_train)

 

    
  
   

    


  


  
