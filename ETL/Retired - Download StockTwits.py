# Databricks notebook source
# MAGIC %md
# MAGIC ## Download Stocktwits Data
# MAGIC * https://notebook.community/tdrussell/stocktwits_analysis/stocktwits_analysis

# COMMAND ----------

import io, json, requests, time, os, os.path, math, urllib
from sys import stdout
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn import linear_model
from pandas_datareader.data import get_data_yahoo
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# COMMAND ----------

# returns python object representation of JSON in response
def get_response(symbol, older_than, retries=5):
  url = f'https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json?max={older_than-1}'
  for _ in range(retries):
      response = requests.get(url)
      if response.status_code == 200:
          return json.loads(response.content)
      elif response.status_code == 429:
          print(response.content)
          return None
      time.sleep(1.0)
  # couldn't get response
  return None

# COMMAND ----------

# extends the current dataset for a given symbol with more tweets
def get_older_tweets(symbol, num_queries):    
  # path = './data/%s.json' % symbol

  if os.path.exists(path):
      # extending an existing json file
      with open(path, 'r') as f:
          data = json.load(f)
          if len(data) > 0:
              older_than = data[-1]['id']
          else:
              older_than = 1000000000000
  else:
      # creating a new json file
      data = []
      older_than = 1000000000000 # any huge number
  
  for i in range(num_queries):
      content = get_response(symbol, older_than)
      if content == None:
          print('Error, an API query timed out')
          break
      data.extend(content['messages'])
      older_than = data[-1]['id']
      stdout.write('\rSuccessfully made query %d' % (i+1))
      stdout.flush()
      # sleep to make sure we don't get throttled
      time.sleep(0.5)
      
  # # write the new data to the JSON file
  # with open(path, 'w') as f:
  #     json.dump(data, f)
  
  return data

# COMMAND ----------

symbol = "T"
num_queries = 1
older_than = 1 # any huge number

content = get_response(symbol, older_than)
# get_older_tweets(symbol, num_queries)

# COMMAND ----------

content

# COMMAND ----------



# COMMAND ----------

retries = 5
symbol = "T"
older_than = 100

url = f'https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json?max={older_than-1}'

for _ in range(retries):
  response = requests.get(url)
  
  if response.status_code == 200:
    print("Good")
    print(json.loads(response.content)) 
  elif response.status_code == 429:
    print("Bad")
    print(response.content)
  else:
    print("What:", response.status_code)

  time.sleep(1.0)

# COMMAND ----------

symbol = "T"
url = f"https://stocktwits.com/symbol/{symbol}"

response = requests.get(url)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


