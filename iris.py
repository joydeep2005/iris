# S10.1: Copy this code cell in 'iris_app.py' using the Sublime text editor. You have already created this ML model in the previous class(es).
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# from annotated_text import annotated_text
# from htbuilder.units import unit
# from htbuilder import H,HtmlElement,styles

# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Loading the dataset.
iris_df = pd.read_csv("iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating a model for Support Vector classification to classify the flower types into labels '0', '1', and '2'.

# Creating features and target DataFrames.
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

# Splitting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Creating the SVC model and storing the accuracy score in a variable 'score'.
a = RandomForestClassifier(n_jobs=-1,n_estimators = 100)

b = LogisticRegression()
a.fit(X_train,y_train)
b.fit(X_train,y_train)
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
score = svc_model.score(X_train, y_train)
s  = a.score(X_train,y_train)
c = b.score(X_train,y_train)


st.title('iris flower prediction')
st.sidebar.subheader('select values')
sw = st.sidebar.slider('SepalWidthCm',float(iris_df['SepalWidthCm'].min()),float(iris_df['SepalWidthCm'].max()))
sl = st.sidebar.slider('SepalLengthCm',float(iris_df['SepalLengthCm'].min()),float(iris_df['SepalLengthCm'].max()))
pl = st.sidebar.slider('PetalLengthCm',float(iris_df['PetalLengthCm'].min()),float(iris_df['PetalLengthCm'].max()))
pw  =st.sidebar.slider('PetalWidthCm',float(iris_df['PetalWidthCm'].min()),float(iris_df['PetalWidthCm'].max()))
o = ['LogisticRegression','RandomForestClassifier','SVC']
z = st.sidebar.radio('select classisfier',o)
def predict(sl,sw,pl,pw,z):
	y_pred = z.predict([[sl,sw,pl,pw]])
	if y_pred[0] ==0:
		return 'Iris Setosa'
	if y_pred[0] ==1:
		return 'Iris-virginica'
	else:
		return 'Iris-versicolor'
     
if st.sidebar.button('predict'):

  if z == 'LogisticRegression':
    pred 	=		predict(sl,sw,pl,pw,b)
    st.write(('flower:',str(pred))
    st.write(('score:',str(c))
  if z == 'RandomForestClassifier':

    pred = predict(sl,sw,pl,pw,a)
    st.info(('flower:',str(pred)))
    st.annotated_text(('score:',' ','#2ae'),str(s))
  if z == 'SVC':
    pred = predict(sl,sw,pl,pw,svc_model)
    st.info(('flower:',str(pred))
    st.success(('score:',' ','#2ae'),str(score))
