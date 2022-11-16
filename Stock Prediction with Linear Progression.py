#!/usr/bin/env python
# coding: utf-8

# ## Stock Prediction Project

# Scrape Data via API. We will use yfinance

# In[1]:


get_ipython().system('pip install yfinance')


# In[6]:


import yfinance as yf
import datetime 
from datetime import date
import matplotlib.pyplot as pyplot
import urllib.request
import json

from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np


# In our case, we will use Walt Disney (NYSE: DIS) as our sample dataset

# In[3]:


stock = input("Please select your stock/share/forex: ")
today = date.today()


# In[7]:


dataframe = yf.download(stock, "2010-01-01", today, auto_adjust=True)
dataframe = dataframe[["Close"]] #only require close for prediction
dataframe = dataframe.dropna()
print(len(dataframe))
dataframe.head()


# In[8]:


def get_yahoo_shortname(symbol):
    response = urllib.request.urlopen(f'https://query2.finance.yahoo.com/v1/finance/search?q={symbol}')
    content = response.read()
    data = json.loads(content.decode('utf8'))['quotes'][0]['shortname']
    return data


# In[9]:


name = get_yahoo_shortname(stock)


# In[10]:


# visualize the dataset
dataframe.Close.plot(figsize = (10, 5), color = "g")
pyplot.ylabel(name+" Stock Value")
pyplot.title(name+" ("+stock+") - 2010 - "+today.strftime("%Y"))


# Define variables

# In[11]:


dataframe["five_days_moving_avg"] = dataframe["Close"].rolling(window=5).mean()
dataframe["twenty_days_moving_avg"] = dataframe["Close"].rolling(window=20).mean()

dataframe = dataframe.dropna()

X = dataframe[["five_days_moving_avg", "twenty_days_moving_avg"]]

dataframe["value_next_day"] = dataframe["Close"].shift(-1)
dataframe = dataframe.dropna()

y = dataframe["value_next_day"]


# Split Dataset into Train and Test

# In[12]:


split_index = 0.8 # Split data into 80:20

split_index = split_index * len(dataframe)

split_index = int(split_index)

X_train = X[:split_index]
y_train = y[:split_index]

X_test = X[split_index:]
y_test = y[split_index:]


# Prepare Linear Regression Model

# In[13]:


model = LinearRegression()
model = model.fit(X_train, y_train)
five_day_moving_avg = model.coef_[0]
twenty_day_moving_avg = model.coef_[1]

print(five_day_moving_avg)
print(twenty_day_moving_avg)


# In[14]:


constant = model.intercept_
print(constant)


# Make prediction on Stock Prices

# In[15]:


test_output = model.predict(X_test)

y_test = y[(split_index - 1):]

test_output = pd.DataFrame(test_output, index= y_test.index, columns = ["value"])

test_output.plot(figsize = (10, 5), color = "g")
y_test.plot(color = "orange")

pyplot.legend(["model output", "actual value"])
pyplot.ylabel(name+" Stock Value")


# Calculate Model Accuracy

# In[16]:


score = model.score(X[split_index:], y[(split_index -1):])

score = score*100

print(score)


# This model are able to achieve 99% of accuracy which show a good pattern

# In[17]:


stocks = pd.DataFrame()
stocks["value"] = dataframe[split_index:]["Close"]

stocks["predicted_tomorrow_value"] = test_output

stocks["actual_tomorrow_value"] = y_test

stocks["returns"] = stocks["value"].pct_change().shift(-1) # pct = percentage

stocks["strategy"] = np.where(stocks.predicted_tomorrow_value.shift(1) < stocks.predicted_tomorrow_value, 1, 0)

stocks["strategy_returns"] = stocks.strategy * stocks["returns"]

cumulative_product = (stocks["strategy_returns"]+1).cumprod()

cumulative_product.plot(figsize = (10, 5))
pyplot.ylabel("Cumulative Returns")


# In[18]:


print(stocks.head())


# Make predictions whether to buy or to sell

# In[20]:


dataset = yf.download(stock, "2010-01-01", today, auto_adjust=True)

dataset["five_days_avg"] = dataset["Close"].rolling(window=5).mean()
dataset["twenty_days_avg"] = dataset["Close"].rolling(window=20).mean()

dataset = dataset.dropna()

dataset["predicted_stock_value"] = model.predict(dataset[["five_days_avg", "twenty_days_avg"]])

dataset["strategy"] = np.where(dataset.predicted_stock_value.shift(1) < dataset.predicted_stock_value, "Buy", "Hold/Sell")

print(dataset)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




