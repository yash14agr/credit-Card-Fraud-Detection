#non-fraud => misc_net	NC	4.97	28654	36.0788	-81.1781	3495	36.011293	-82.048315	35	0	1	1
# Fraud    => grocery_pos	NC	281.06	28611	35.9946	-81.7266	885	36.430124	-81.179483	35	1	2	1

import pickle
import streamlit as st
import pandas as pd
#import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

cat=['misc_net', 'grocery_pos', 'entertainment', 'gas_transport',
       'misc_pos', 'grocery_net', 'shopping_net', 'shopping_pos',
       'food_dining', 'personal_care', 'health_fitness', 'travel',
       'kids_pets', 'home']

state =['NC', 'WA', 'ID', 'MT', 'VA', 'PA', 'KS', 'TN', 'IA', 'WV', 'FL',
       'CA', 'NM', 'NJ', 'OK', 'IN', 'MA', 'TX', 'WI', 'MI', 'WY', 'HI',
       'NE', 'OR', 'LA', 'DC', 'KY', 'NY', 'MS', 'UT', 'AL', 'AR', 'MD',
       'GA', 'ME', 'AZ', 'MN', 'OH', 'CO', 'VT', 'MO', 'SC', 'NV', 'IL',
       'NH', 'SD', 'AK', 'ND', 'CT', 'RI']

Algo=['Logistic Regression','Random Forest','AutoEncoder', 'Ensemble Learning', 'SVC']



pipe1 = pickle.load(open('pipe1.pkl','rb'))     #logistic regression
pipe2 = pickle.load(open('pipe2.pkl','rb'))     #Random Forest
pipe4 = pickle.load(open('pipe4.pkl','rb'))     #SVC
ensemble = pickle.load(open('ensemble.pkl','rb'))   #Ensemble using above 3 model
preprocessor = pickle.load(open('preprocessor.pkl','rb'))   
AutoEncoder_model =load_model('AutoEncoder.h5') #AutoEncoder
st.title('Credit Card Fraud Detection')

col1, col2 = st.columns(2)
col3,col4=st.columns(2)
col5, col6= st.columns(2)
col7, col8= st.columns(2)
col9, col10= st.columns(2)
col11, col12= st.columns(2)
#category	state	amt	zip	lat	long	city_pop	merch_lat	merch_long	age	hour	day	month
with col1:
    Cat = st.selectbox('Select Category',sorted(cat))
with col2:
    state = st.selectbox('Select State',sorted(state))
with col3:
    amt = st.number_input('Select Amount',format="%.5f")
with col4:
    zip = st.number_input('Select Zip')
with col5:
    lat = st.number_input('Latitude ',format="%.5f")
with col6:
    long = st.number_input('Longitude',format="%.5f")
with col7:
    city_pop = st.number_input('city population')
with col8:
    merch_lat = st.number_input('merch_latitude',format="%.5f")
with col9:
    merch_long = st.number_input('merch_longitude',format="%.5f")
with col10:
    age = st.number_input('Select Age')
with col11:
    hour = st.number_input('Select Hour')
with col12:
    day = st.number_input('day')
    
month = st.number_input('month')


    
# category	state	amt	zip	lat	long	city_pop	merch_lat	merch_long	age	hour	day	month
input_df = pd.DataFrame({
        'category':[Cat],
        'state': [state],
        'amt': [amt],
        'zip': [zip],
        'lat': [lat],
        'long': [long],
        'city_pop':[city_pop],
        'merch_lat': [merch_lat],
        'merch_long': [merch_long],
        'age': [age],
        'hour':[hour],
        'day': [day],
        'month': [month]
    })


algo = st.selectbox('Select Algorithm',Algo)
st.text('you selected : '+algo)

st.table(input_df)

if st.button('Predict :'):
    result=0
    if algo ==Algo[0]:
        result = pipe1.predict(input_df)
    elif algo == Algo[1]:
        result = pipe2.predict(input_df)
    elif algo == Algo[2]:
        data = preprocessor.transform(input_df)
        result = AutoEncoder_model.predict(data)
    elif algo == Algo[3]:
        result = ensemble.predict(input_df)
    elif algo == Algo[4]:
        result = pipe4.predict(input_df)
    # st.text(result)

    if result == 0 or result <= .5 :
        st.text('Transaction is not a fraud Transaction !')
    else:
        st.text('Transaction is a fraud Transaction !')
