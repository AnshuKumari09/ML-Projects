import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Header ko ignore karo, column names khud bana do
column_names = [str(i) for i in range(60)] + ['Label']  # 60 features + 1 label

# CSV load karo
solar_data = pd.read_csv('Copy-of-sonar-data.csv', header=None, names=column_names)
solar_data.groupby('Label').mean()
solar_data.describe()
X=solar_data.drop(columns='Label',axis=1)
df=solar_data.copy()
y=df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,stratify=y, random_state=1)
model = LogisticRegression()
model.fit(X_train,y_train)
training_prediction = model.predict(X_train)
print(accuracy_score(training_prediction,y_train))
test_prediction = model.predict(X_test)
print(accuracy_score(test_prediction,y_test))

st.title("Sonar Rock VS Mine Prediction")
input_data = st.text_input('Enter comma-separated values here')

# Predict and show result on button click
if st.button('Predict'):
    # Prepare input data
    input_data_np_array = np.asarray(input_data.split(','), dtype=float)
    reshaped_input = input_data_np_array.reshape(1, -1)
    # Predict and show result
    prediction = model.predict(reshaped_input)
    if prediction[0] == 'R':
        st.write('This Object is Rock')
    else:
        st.write('The Object is Mine')