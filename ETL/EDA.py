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

# MAGIC %sql
# MAGIC Create table if not exists `31184_cerebro_prd`.`cv0361`.`select_stock`(
# MAGIC   Symbol string,
# MAGIC   Downloaded date,
# MAGIC   ErrorMsg string
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC Insert into `31184_cerebro_prd`.`cv0361`.`select_stock`(Symbol)
# MAGIC values('NVDA')

# COMMAND ----------

# %sql
# Update `31184_cerebro_prd`.`cv0361`.`select_stock`
# Set Downloaded = current_date()
# Where Symbol in ('AAPL', 'TSLA', 'T')

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

# MAGIC %sql
# MAGIC create table if not exists `31184_cerebro_prd`.`cv0361`.`company` 
# MAGIC as
# MAGIC SELECT Symbol, replace(replace(replace(Name, ' Common Stock', ''), ' Common Shares', ''), ' Ordinary Shares', '') as Name
# MAGIC FROM `31184_cerebro_prd`.`cv0361`.`stock`
# MAGIC Where Name not like '% Warrants' AND Name not like '% Warrant';

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * 
# MAGIC FROM `31184_cerebro_prd`.`cv0361`.`company`
# MAGIC Where Symbol = 'T';

# COMMAND ----------


