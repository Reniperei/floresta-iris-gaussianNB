import streamlit as st
from sklearn.naive_bayes import GaussianNB
dados=pd.read_csv('Iris -Iris.csv')
st.title('naive bayes IA iris')
st.write ('criando IA')

import pandas as pd
import numpy as np
  
X = iris.data[:4]
y = iris.target[:4]
print("X:", X)
print("Y:",y)

# splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=4)
  
# training the model on training set
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
  
# making predictions on the testing set
y_pred = gnb.predict(X_test)
  
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

 

    
  
   

    


  


  
