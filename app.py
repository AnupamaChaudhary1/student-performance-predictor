import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import streamlit as st

#stramlit congigs
st.set_page_config(page_title="Student Performance Predictor", layout='centered')
st.title('Student Performance Predictor')
st.write('This web app use a simple machine learning model to predict a students average score based on input subject marks')
maths=st.slider('Maths Score', 0,100,50)
science=st.slider('Science Score', 0,100,50)
english=st.slider('English Score', 0,100,50)


training_data=pd.DataFrame({
    'maths':np.random.randint(30,100,50),
    'science':np.random.randint(30,100,50),
    'english':np.random.randint(30,100,50)
})

#Average
training_data['average']=training_data.mean(axis=1)
X=training_data[['maths','science','english']]
y=training_data['average']
#spllit data
train_X,test_X, train_y,test_y=train_test_split(X,y,test_size=0.2)

#train model

model=LinearRegression()
model.fit(X,y)

new_student= pd.DataFrame({
    'maths':[85],
    'science':[76],
    'english':[80]
})

input_data=[[maths,science,english]]
prediction=model.predict(input_data)[0]

st.success(f"Predicted Score is: {round(prediction,2)}")


# Visualization
#plotting training data and prediction
st.subheader('Visual comparision')
fig,ax=plt.subplots()
ax.scatter(training_data['maths'],training_data['average'],color='blue',label='Training Data')
ax.scatter(maths,prediction,color='red',label='Input',s=100)
ax.set_xlabel('Maths Score')
ax.set_ylabel('Predicted Average')
ax.set_title('Maths vs agerage prediction')
ax.legend()

st.pyplot(fig)


# ax.scatter(training_data['science'],training_data['average'],color='blue',label='Training Data')
# ax.scatter(science,prediction,color='red',label='Input',s=100)
# ax.scatter(training_data['english'],training_data['average'],color='blue',label='Training Data')
# ax.scatter(english,prediction,color='red',label='Input',s=100)