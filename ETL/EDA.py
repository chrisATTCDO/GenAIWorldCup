# Databricks notebook source
import pandas as pd

# COMMAND ----------

# df = (spark.read
#   .format("csv")
#   .option("mode", "PERMISSIVE")
#   .option("header", "true")
#   .load("/Volumes/31184_cerebro_prd/cv0361/stockguru/Stock.csv")
# )

# df.display()

# COMMAND ----------

# pd_df = df.toPandas()
# pd_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Stocks Selection Table

# COMMAND ----------

# %sql
# Create table if not exists `31184_cerebro_prd`.`cv0361`.`select_stock`(
#   Symbol string,
#   Downloaded date,
#   ErrorMsg string
# )

# COMMAND ----------

# %sql
# Insert into `31184_cerebro_prd`.`cv0361`.`select_stock`(Symbol)
# values('AMD')

# COMMAND ----------

# %sql
# Update `31184_cerebro_prd`.`cv0361`.`select_stock`
# Set Downloaded = current_date()
# Where Symbol in ('AAPL', 'TSLA', 'T')

# COMMAND ----------

# MAGIC %sql
# MAGIC Update `31184_cerebro_prd`.`cv0361`.`select_stock`
# MAGIC Set Downloaded = null
# MAGIC -- Where Symbol not in ('AMD','HD','KO','PLTR','SBUX','UBER','WMT')

# COMMAND ----------

# MAGIC %sql
# MAGIC Select *
# MAGIC From `31184_cerebro_prd`.`cv0361`.`select_stock`
# MAGIC -- Where Downloaded is null 
# MAGIC --   OR datediff(DAY, Downloaded, current_date()) > 0
# MAGIC ORDER BY Downloaded, Symbol

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Company Table from Stock

# COMMAND ----------

# %sql
# create table if not exists `31184_cerebro_prd`.`cv0361`.`company` 
# as
# SELECT Symbol, replace(replace(replace(Name, ' Common Stock', ''), ' Common Shares', ''), ' Ordinary Shares', '') as Name
# FROM `31184_cerebro_prd`.`cv0361`.`stock`
# Where Name not like '% Warrants' AND Name not like '% Warrant';

# COMMAND ----------

# %sql
# SELECT * 
# FROM `31184_cerebro_prd`.`cv0361`.`company`
# Where Symbol = 'T';

# COMMAND ----------



# COMMAND ----------

# import os

# os.getenv("no_proxy")

# COMMAND ----------


