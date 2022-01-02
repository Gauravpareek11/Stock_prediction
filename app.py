import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.python.keras.saving.save import load_model
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import pickle
from tqdm.notebook import tnrange

class MultiDimensionScaler():
  def __init__(self):
    self.scalers=[]
  def fit_transform(self,X):
    total_dims=X.shape[2]
    for i in range(total_dims):
      Scaler=MinMaxScaler()
      X[:,:,i]=Scaler.fit_transform(X[:,:,i])
      self.scalers.append(Scaler)
    return X
  def transform(self,X):
    for i in range(X.shape[2]):
      X[:,:,i]=self.scalers[i].transform(X[:,:,i])
    return X
def CreateFeautres_and_Targets(data,feautre_length):
  x=[]
  y=[]
  for i in tnrange(len(data)-feautre_length):
    x.append(data.iloc[i:i+feautre_length:].values)
    y.append(data['Close'].values[i+feautre_length])

  x=np.array(x)
  y=np.array(y)
  return x,y

def predictStock(model,DataFrame,previousDate,feautre_length=32):
  idx_location=DataFrame.index.get_loc(previousDate)
  Feautres=DataFrame.iloc[idx_location-feautre_length:idx_location,:].values
  Feautres=np.expand_dims(Feautres,axis=0)
  Feautres=Feautre_Scaler.transform(Feautres)
  predictions=model.predict(Feautres)
  predictions=Target_Scaler.inverse_transform(predictions)
  return predictions[0][0]

def save_object(obj,name:str):
  pickle_out=open(f"{name}.pck","wb")
  pickle.dump(obj,pickle_out)
  pickle_out.close()

def load_object(name:str):
  pickle_in=open(f"{name}.pck","rb")
  data=pickle.load(pickle_in)
  return data

st.title('Stock_Price_Prediction')

a=st.text_input('Enter Stock Ticker',"TATAMOTORS.NS")
st.write('Search your stock from here')
st.write('https://finance.yahoo.com')
data=yf.download(a,start="2018-01-01",interval='1d')
data.sort_index(inplace=True)
data=data.loc[~data.index.duplicated(keep='first')] 

st.subheader('Data from 2018-till Date')
st.write(data.describe())

st.subheader('Closing Price Vs Time')
fig=go.Figure()
fig.add_trace(go.Scatter(x=data.index,y=data['Close'],mode='lines'))
fig.update_layout(height=500,width=900,xaxis_title='Date',yaxis_title='close')
st.plotly_chart(fig)


st.subheader('Volume Vs Time')
fig=go.Figure()
fig.add_trace(go.Scatter(x=data.index,y=data['Volume'],mode='lines'))
fig.update_layout(height=500,width=900,xaxis_title='Date',yaxis_title='Volume')
st.plotly_chart(fig)

st.subheader('100 days Moving Average')
ma100=data['Close'].rolling(100).mean()
fig=go.Figure()
fig.add_trace(go.Scatter(x=data.index,y=data['Close'],mode='lines',name='closing'))
fig.add_trace(go.Scatter(x=data.index,y=ma100,mode='lines',name='Moving_avg_100'))
fig.update_layout(height=500,width=900,xaxis_title='Date',yaxis_title='close')
st.plotly_chart(fig)

st.subheader('200 days Moving Average')
ma200=data['Close'].rolling(200).mean()
fig=go.Figure()
fig.add_trace(go.Scatter(x=data.index,y=data['Close'],mode='lines',name='closing'))
fig.add_trace(go.Scatter(x=data.index,y=ma100,mode='lines',name='Moving_avg_100'))
fig.add_trace(go.Scatter(x=data.index,y=ma200,mode='lines',name='Moving_avg_200'))
fig.update_layout(height=500,width=900,xaxis_title='Date',yaxis_title='close')
st.plotly_chart(fig)


data=data[['Close','Volume']]
test_length=data[(data.index>="2021-01-01")].shape[0]
x,y=CreateFeautres_and_Targets(data,32)
Xtrain,Xtest,Ytrain,Ytest=x[:-test_length],x[-test_length:],y[:-test_length],y[-test_length:]

Feautre_Scaler=MultiDimensionScaler()
Xtrain=Feautre_Scaler.fit_transform(Xtrain)
Xtest=Feautre_Scaler.fit_transform(Xtest)

Target_Scaler=MinMaxScaler()
Ytrain=Target_Scaler.fit_transform(Ytrain.reshape(-1,1))
Ytest=Target_Scaler.fit_transform(Ytest.reshape(-1,1))

save_object(Feautre_Scaler,"Feautre_scaler")
save_object(Target_Scaler,"Target_scaler")

model=load_model('stock.h5')

predictions=model.predict(Xtest)
predictions=Target_Scaler.inverse_transform(predictions)
Actual=Target_Scaler.inverse_transform(Ytest)
predictions=np.squeeze(predictions,axis=1)
Actual=np.squeeze(Actual,axis=1)

st.subheader('Predicted Vs Actual')
fig=go.Figure()
fig.add_trace(go.Scatter(x=data.index[-test_length:],y=predictions,mode='lines',name='predicted'))
fig.add_trace(go.Scatter(x=data.index[-test_length:],y=Actual,mode='lines',name='Actual'))
fig.update_layout(height=500,width=900,xaxis_title='Date',yaxis_title='Price')
st.plotly_chart(fig)

from datetime import date
from datetime import timedelta
st.subheader('Next Day Predicted Price')
try:
  dat=date.today()- timedelta(days = 1)
  # st.write(dat)
  st.write(predictStock(model,data,str(dat)))
except:
  try:
    dat=date.today()- timedelta(days = 2)
    # st.write(dat)
    # st.subheader('Next Day Predicted Price')
    st.write(predictStock(model,data,str(dat)))
  except:
    dat=date.today()- timedelta(days = 3)
    # st.write(dat)
    # st.subheader('Next Day Predicted Price')
    st.write(predictStock(model,data,str(dat)))


