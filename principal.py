import streamlit as st

st.title('naive bayes IA iris')
st.write ('criando IA')

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


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

# Import function to fetch dataset
from sklearn.datasets import load_iris

# Import Multinomial Naive-Bayes classifier
from sklearn.naive_bayes import GaussianNB

# Import train_test_split
from sklearn.model_selection import train_test_split

# Import sklearn metrics for analysis
from sklearn.metrics import classification_report, confusion_matrix

# Import heatmap plotting function
from matrix_heatmap import matrix_heatmap

# Import custom latex display and numbering class
from latex_equation_numbering import latex_equation_numbering

# Load helpers
from helpers import button_created, button_changed


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
 
  run_button = st.button('Run Code', key=run_button_key, on_click=button_created(run_button_key))
  st.subheader('Output:')
  output_col1, output_col2 = st.beta_columns(2)

  if run_button or st.session_state[run_button_key+'_dict']['was_pressed']:
  st.session_state[run_button_key+'_dict']['was_pressed'] = True

  iris_df = load_iris(as_frame=True)
  iris_df.target.replace({0: 'setosa', 1: 'versicolor', 2: 'virginica'}, inplace=True)

  iris_features_train, iris_features_test, iris_species_train, iris_species_test \
   = train_test_split(iris_df.data, iris_df.target, train_size=0.6, shuffle=True, random_state=42)
        
  classifier = GaussianNB()

  classifier.fit(iris_features_train, iris_species_train)

  iris_species_predict = classifier.predict(iris_features_test)

  # Create confusion matrix DataFrame
  cm_df = pd.DataFrame(data=confusion_matrix(iris_species_predict, iris_species_test), columns=iris_df.target_names, index=iris_df.target_names)

  # Make a heatmap of the confusion matrix
  fig, ax = plt.subplots()
  fig = matrix_heatmap(cm_df.values.tolist(), options={'x_labels': iris_df.target_names,'y_labels': iris_df.target_names, 'annotation_format': '.3g', 'color_map': 'bone_r', 'custom_range': False, 'vmin_vmax': (-1,1), 'center': None, 'title_axis_labels': ('Confusion matrix heatmap', 'Species', 'Species'), 'rotate x_tick_labels': True})
   
  if output_col1:
  st.write(f'**Classifier accuracy:** {classifier.score(iris_features_test, iris_species_test)}')

  st.write('**Classification report:**')
  st.text('.  \n'+classification_report(iris_species_predict, iris_species_test))

  st.write('**Confusion matrix:**')
  st.write(cm_df)

  if output_col2:              
  st.pyplot(fig)
  st.subheader('')

  st.header('Tuning the model')
    
  
   

    


  


  
