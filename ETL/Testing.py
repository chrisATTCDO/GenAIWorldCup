# Databricks notebook source
# MAGIC %pip install dbdemos

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import dbdemos
dbdemos.install('llm-tools-functions')

# COMMAND ----------



# COMMAND ----------

import requests
import json

# def get_stock_price(symbol):
#     """get a stock price from yahoo finance"""

#     url = "https://query1.finance.yahoo.com/v7/finance/quote?symbols=" + symbol
#     headers = {'User-Agent': 'Mozilla/5.0'}
#     response = requests.get(url, headers=headers)    
#     data = json.loads(response.text)
    
#     return data['quoteResponse']['result'][0]['regularMarketPrice']


# print(get_stock_price('AAPL'))


# COMMAND ----------

symbol = "T"
url = "https://query1.finance.yahoo.com/v7/finance/quote?symbols=" + symbol
headers = {'User-Agent': 'Mozilla/5.0'}

response = requests.get(url, headers=headers)    
data = json.loads(response.text)

# COMMAND ----------

data

# COMMAND ----------


