# Databricks notebook source
# %pip install --quiet -U databricks-sdk==0.23.0 langchain-community==0.2.10 langchain-openai==0.1.19 mlflow==2.14.3 faker
# dbutils.library.restartPython()

# COMMAND ----------

# %run ./_resources/00-init $reset_all=false

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

print(f"USE CATALOG `{catalog}`")
spark.sql(f"USE CATALOG `{catalog}`")

print(f"using catalog.database `{catalog}`.`{db}`")
spark.sql(f"""USE `{catalog}`.`{db}`""")    


# COMMAND ----------

# MAGIC %md
# MAGIC ### Stock Ordering

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION execute_stock_order(
# MAGIC   AccountId string,
# MAGIC   OrderType string,
# MAGIC   Symbol string,
# MAGIC   Price double,
# MAGIC   Quantity int
# MAGIC )
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON
# MAGIC COMMENT 'This function executes stock ordering.'
# MAGIC AS
# MAGIC $$
# MAGIC   try:
# MAGIC     import uuid
# MAGIC     
# MAGIC     TransactionId = str(uuid.uuid4())
# MAGIC
# MAGIC     return TransactionId
# MAGIC   except Exception as e:
# MAGIC     return str(e)
# MAGIC $$;
# MAGIC

# COMMAND ----------

# %sql
# CREATE OR REPLACE FUNCTION execute_stock_order(
#   AccountId string,
#   Symbol string,
#   Price double,
#   Quantity int
# )
# RETURNS STRING
# LANGUAGE PYTHON
# COMMENT 'This function executes stock ordering.'
# AS
# $$
#   try:
#     import uuid
#     from pyspark.sql import SparkSession

#     TransactionId = str(uuid.uuid4())
    
#     spark = SparkSession.builder.getOrCreate()

#     spark.sql(f"INSERT INTO order_stock (TransactionId, AccountId, Symbol, Price, Quantity) VALUES ('{TransactionId}', '{AccountId}', '{Symbol}', {Price}, {Quantity})")

#     return TransactionId
#   except Exception as e:
#     return str(e)
# $$;


# COMMAND ----------

# %sql
# CREATE OR REPLACE FUNCTION execute_stock_order(
#   AccountId string,
#   Symbol string,
#   Price double,
#   Quantity int
# )
# RETURNS STRING
# LANGUAGE PYTHON
# COMMENT 'This function executes stock ordering.'
# AS
# $$
#   try:
#     import uuid
#     from pyspark.shell import spark

#     TransactionId = str(uuid.uuid4())

#     spark.sql(f"INSERT INTO order_stock (TransactionId, AccountId, Symbol, Price, Quantity) VALUES ('{TransactionId}', '{AccountId}', '{Symbol}', {Price}, {Quantity})")

#     return TransactionId
#   except Exception as e:
#     return str(e)
# $$;


# COMMAND ----------

# MAGIC %sql
# MAGIC -- let's test our function:
# MAGIC SELECT execute_stock_order('23456', 'BUY', 'T', 21.95, 100) as TransactionId;
# MAGIC -- SELECT execute_stock_order('23456', 'SELL', 'T', 21.95, 100) as TransactionId;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Stock Transactions

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION get_stock_orders(
# MAGIC                   AccountId STRING,
# MAGIC                   RecordCount INT DEFAULT 3
# MAGIC                 )
# MAGIC RETURNS TABLE(
# MAGIC                 OrderType STRING,
# MAGIC                 Stock STRING,
# MAGIC                 Price DOUBLE,
# MAGIC                 Quantity INT,
# MAGIC                 Amount DOUBLE,
# MAGIC                 TransactionId STRING,
# MAGIC                 OrderDatetime TIMESTAMP
# MAGIC                 )
# MAGIC COMMENT 'Returns a list of customer orders for the given customer ID (expect a UUID)'
# MAGIC LANGUAGE SQL
# MAGIC     RETURN
# MAGIC     Select OrderType, Symbol as Stock, Price, Quantity, Price*Quantity as Amount, TransactionId, OrderDatetime
# MAGIC     From `order_stock`
# MAGIC     WHERE AccountId = get_stock_orders.AccountId
# MAGIC     Order By OrderDatetime Desc
# MAGIC     limit 3;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM get_stock_orders('23456');

# COMMAND ----------


