# Databricks notebook source
# MAGIC %md
# MAGIC # Download Stocks Data

# COMMAND ----------

# %sql
# Select Symbol
# From `31184_cerebro_prd`.`cv0361`.`select_stock`
# -- Where Downloaded is null 
# --   OR datediff(DAY, Downloaded, current_date()) > 0
# ORDER BY Downloaded

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get a List of Stocks to Download

# COMMAND ----------

df = spark.sql("""
Select Symbol
From `31184_cerebro_prd`.`cv0361`.`select_stock`
Where Downloaded is null 
  OR datediff(DAY, Downloaded, current_date()) > 0
ORDER BY Downloaded
""")

# df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loop Thru Stock table and download data for earch applicable symbol

# COMMAND ----------

# symbol = "META"
# spark.sql(f"""Update `31184_cerebro_prd`.`cv0361`.`select_stock` Set Downloaded = current_date() where symbol = '{symbol}'""")

# COMMAND ----------

timeout_seconds = 600 # 10 minutes

for row in df.rdd.collect():
  symbol = row['Symbol']

  print(f"Processing: {symbol}")

  try:
    # Execute notebook to download stock info
    dbutils.notebook.run("Download Data Per Symbol", timeout_seconds, {"Stock_Symbol": symbol})

    # Update Status table set downloaded = current_date where symbol = {symbol}
    spark.sql(f"Update `31184_cerebro_prd`.`cv0361`.`select_stock` Set Downloaded = current_date() where symbol = '{symbol}'")
  except Exception as e:
    print(e)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visually Inspect the Generated Files

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/mnt/regression_testing/hackathon_files/pending"))

# COMMAND ----------


