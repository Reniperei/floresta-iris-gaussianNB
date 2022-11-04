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

from sklearn.datasets import load_iris
iris = load_iris()
  
# store the feature matrix (X) and response vector (y)
X = iris.data[:6]
y = iris.target[:6]
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



  


  
