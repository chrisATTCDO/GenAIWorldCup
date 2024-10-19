# Databricks notebook source
# MAGIC %md
# MAGIC # Download Realtime Stocks Data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get a List of Stocks to Download

# COMMAND ----------

# Only pull realtime data for stock that is behind/current on the downloaded data. 
# NOTE: Set Downloaded to future date to prevent excessive download, which caused the rate limit to exceed.

df = spark.sql("""
Select Symbol
From `31184_cerebro_prd`.`cv0361`.`select_stock`
Where datediff(DAY, Downloaded, current_date()) >= 0
ORDER BY Downloaded
""")

# df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loop Thru Stock table and download data for earch applicable symbol

# COMMAND ----------

timeout_seconds = 600 # 10 minutes

for row in df.rdd.collect():
  symbol = row['Symbol']

  print(f"Processing: {symbol}")

  try:
    # Execute notebook to download stock info
    dbutils.notebook.run("Download Realtime Data Per Symbol", timeout_seconds, {"Stock_Symbol": symbol})
  except Exception as e:
    print(e)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visually Inspect the Generated Files

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/mnt/regression_testing/hackathon_files/pending"))

# COMMAND ----------


