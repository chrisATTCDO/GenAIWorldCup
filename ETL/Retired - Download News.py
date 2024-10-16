# Databricks notebook source
# MAGIC %md
# MAGIC ## Download News

# COMMAND ----------

import io, json, requests, time, os
from bs4 import BeautifulSoup

# COMMAND ----------

symbol = "T"
url = f"https://finance.yahoo.com/news/goldman-sachs-adjusts-price-target-110354739.html"
print(url)

response = requests.get(url)

# COMMAND ----------

response.content

# COMMAND ----------

soup = BeautifulSoup(response.content)

# COMMAND ----------

print(soup.prettify())

# COMMAND ----------


