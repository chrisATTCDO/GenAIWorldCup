# Databricks notebook source
import pandas as pd

# COMMAND ----------

# df = pd.read_csv("/Volumes/31184_cerebro_prd/cv0361/stockguru/Stock.csv")
# df.head()

# COMMAND ----------

df = (spark.read
  .format("csv")
  .option("mode", "PERMISSIVE")
  .option("header", "true")
  .load("/Volumes/31184_cerebro_prd/cv0361/stockguru/Stock.csv")
)

df.display()

# COMMAND ----------

pd_df = df.toPandas()
pd_df.head()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * 
# MAGIC FROM `31184_cerebro_prd`.`cv0361`.`company`;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * 
# MAGIC FROM `31184_cerebro_prd`.`cv0361`.`company`
# MAGIC Where Symbol = 'T';

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * 
# MAGIC FROM `31184_cerebro_prd`.`cv0361`.`company`
# MAGIC Where Name not like '% Warrants' or Name not like '% Warrant';

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT Symbol, replace(replace(replace(Name, ' Common Stock', ''), ' Common Shares', ''), ' Ordinary Shares', '') as Name
# MAGIC FROM `31184_cerebro_prd`.`cv0361`.`company`
# MAGIC Where Name not like '% Warrants' AND Name not like '% Warrant';

# COMMAND ----------

# MAGIC %sql
# MAGIC create table if not exists `31184_cerebro_prd`.`cv0361`.`test` 
# MAGIC as
# MAGIC SELECT Symbol, replace(replace(replace(Name, ' Common Stock', ''), ' Common Shares', ''), ' Ordinary Shares', '') as Name
# MAGIC FROM `31184_cerebro_prd`.`cv0361`.`company`
# MAGIC Where Name not like '% Warrants' AND Name not like '% Warrant';

# COMMAND ----------


