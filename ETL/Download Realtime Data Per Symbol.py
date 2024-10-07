# Databricks notebook source
# MAGIC %md
# MAGIC # Download Realtime Data Per Stock Symbol

# COMMAND ----------

dbutils.widgets.text("Stock_Symbol", "T")

# COMMAND ----------

import yfinance as yf
import json
from pprint import pprint
from time import strftime, localtime
import datetime

# COMMAND ----------

Symbol = dbutils.widgets.get("Stock_Symbol").strip().upper()

if Symbol == "":
  dbutils.notebook.exit("Stock Symbol is required!")

print("Symbol:", Symbol)

File_Path = '/dbfs/mnt/regression_testing/hackathon_files/pending/'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Functions

# COMMAND ----------

def get_value(obj, key):
  try:
    return obj[key]
  except:
    return ''

# COMMAND ----------

def epoch_to_date(epoch_time):
  if epoch_time == '':
      return ''
    
  return strftime('%m/%d/%Y %H:%M:%S', localtime(epoch_time))

# COMMAND ----------

def write_stock_info(file_name, header, my_dict):
  with open(file_name, 'w') as file:
    file.write(header + "\n")
    file.write(json.dumps(my_dict)) 


# COMMAND ----------

def show_file_content(filename, MAX_LINES=5):
  count = 1
  
  with open(filename, "r") as f:
    lines = f.readlines()
    print(f"{filename}\tLines: {len(lines)}\n========================================================================================")
    for line in lines:
      print(line) 

      count += 1
      if count > MAX_LINES:
        break

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download Data

# COMMAND ----------

ticker = yf.Ticker(Symbol)

# COMMAND ----------

company_header = "Company: " + ticker.info['longName'] + "\nStock Symbol: " + Symbol

print(company_header)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Realtime Quote Placeholder
# MAGIC * Generate placeholder values... Realtime Job will overwrite this values

# COMMAND ----------

realtime = {
  'Current Price': ticker.info['open'],
  'day open Price': ticker.info['open'],
  'day Low Price': ticker.info['dayLow'],
  'day High Price': ticker.info['dayHigh'],
  'bid Size': ticker.info['bidSize'],
  'ask Size': ticker.info['askSize'],
  'Stock Volume': ticker.info['volume'],
}

file_name = f"{File_Path}{Symbol}_realtime.txt"
write_stock_info(file_name, company_header + "\nStock Realtime Quote JSON Format. Refresh every 15-minutes.", realtime)
show_file_content(file_name)

# COMMAND ----------

print("Notebook Execution Completed:", datetime.datetime.now())

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/mnt/regression_testing/hackathon_files/pending"))

# COMMAND ----------


