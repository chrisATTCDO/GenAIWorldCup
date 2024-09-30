# Databricks notebook source
# MAGIC %md
# MAGIC ## Try Something New
# MAGIC * Shared Storage: /dbfs/mnt/regression_testing/hackathon_files/pending

# COMMAND ----------

import json
from pprint import pprint

# COMMAND ----------

print("Hi, hello world")

# COMMAND ----------

display(dbutils.fs.ls(f"dbfs:/mnt/regression_testing/hackathon_files/pending"))

# COMMAND ----------

def show_file_content(filename, MAX_LINES=5):
  count = 1
  
  with open(filename, "r") as f:
    lines = f.readlines()
    print(f"Lines Count: {len(lines)}\n==================================")
    for line in lines:
      print(line) 

      count += 1
      if count > MAX_LINES:
        break

# COMMAND ----------

show_file_content("/dbfs/mnt/regression_testing/hackathon_files/pending/StockAT&TJun24_1.csv")

# COMMAND ----------

# df.write.mode("overwrite").option("header", "true").format("com.databricks.spark.csv").save("dbfs:/mnt/regression_testing/hackathon_files/pending/test.csv")

# COMMAND ----------

file_name = '/dbfs/mnt/regression_testing/hackathon_files/pending/testing.txt'

# COMMAND ----------

with open(file_name, 'w') as file_out:
  file_out.write("Testing")

# COMMAND ----------

show_file_content(file_name)

# COMMAND ----------

my_dict = {
    'key1': 'value1',
    'key2': 1234,
    'my key': 'my value'
  }

with open(file_name, 'w') as file:
  file.write(json.dumps(my_dict)) 


# COMMAND ----------

with open(file_name, "r") as file:
  my_json = json.load(file)

my_json

# COMMAND ----------

pprint(my_json, compact=True)

# COMMAND ----------

my_json["my key"]

# COMMAND ----------


