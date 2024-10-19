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

# %sql
# DELETE From `31184_cerebro_prd`.`cv0361`.`select_stock`
# Where Symbol = 'RF'

# COMMAND ----------

# MAGIC %sql
# MAGIC Update `31184_cerebro_prd`.`cv0361`.`select_stock`
# MAGIC Set Downloaded = current_date()
# MAGIC Where Symbol in ('AMD','T','VZ','TMUS','NFLX','PLTR','APPL','AMZN','META','GOOGL','NVDA','TSLA','MSFT','INTC')

# COMMAND ----------

# %sql
# Select distinct sector
# from `31184_cerebro_prd`.`cv0361`.`stock`

# COMMAND ----------

# %sql
# Insert into `31184_cerebro_prd`.`cv0361`.`select_stock`(Symbol)
# Select Symbol
# FROM `31184_cerebro_prd`.`cv0361`.`stock`
# WHERE Country = 'United States'
#   and Sector in ('Technology', 'Telecommunications', 'Health Care', 'Finance', 'Energy')
#   and Volume > 10000000
#   and Name like '%Common Stock'
#   and Symbol not in (Select Symbol FROM `31184_cerebro_prd`.`cv0361`.`select_stock`)

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

# MAGIC %md
# MAGIC ## Create Tables for Stock Order Transaction

# COMMAND ----------

# MAGIC %sql
# MAGIC Create table if not exists `31184_cerebro_prd`.`cv0361`.`customer`(
# MAGIC   AccountId string,
# MAGIC   LoginId string,
# MAGIC   FirstName string,
# MAGIC   LastName string,
# MAGIC   Email string,
# MAGIC   JoinedDate  date
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC -- drop table `31184_cerebro_prd`.`cv0361`.`order_stock`;
# MAGIC
# MAGIC Create table if not exists `31184_cerebro_prd`.`cv0361`.`order_stock`(
# MAGIC   TransactionId string,
# MAGIC   AccountId string,
# MAGIC   OrderType string,
# MAGIC   Symbol string,
# MAGIC   Price double,
# MAGIC   Quantity int,
# MAGIC   OrderDatetime timestamp
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Insert into `31184_cerebro_prd`.`cv0361`.`customer`(AccountId, LoginId, FirstName, LastName, Email, JoinedDate)
# MAGIC -- values('12345', 'maustin', 'Mark', 'Austin', 'ma2552@att.com', '2020-01-01');
# MAGIC -- Insert into `31184_cerebro_prd`.`cv0361`.`customer`(AccountId, LoginId, FirstName, LastName, Email, JoinedDate)
# MAGIC -- values('23456', 'chriscdo', 'Chris', 'Vo', 'cv0361@att.com', '2024-01-01');
# MAGIC -- Insert into `31184_cerebro_prd`.`cv0361`.`customer`(AccountId, LoginId, FirstName, LastName, Email, JoinedDate)
# MAGIC -- values('34567', 'pavantagirisa', 'Pavan', 'Tagirisa', 'pt828n@att.com', '2020-05-03');
# MAGIC -- Insert into `31184_cerebro_prd`.`cv0361`.`customer`(AccountId, LoginId, FirstName, LastName, Email, JoinedDate)
# MAGIC -- values('45678', 'jagdeesh_infosys', 'Jagdeesh', 'Rajamani', 'ju0929@att.com', '2021-09-22');
# MAGIC
# MAGIC Select *
# MAGIC From `31184_cerebro_prd`.`cv0361`.`customer`

# COMMAND ----------

# %sql
# delete from `31184_cerebro_prd`.`cv0361`.`order_stock`

# COMMAND ----------

# MAGIC %sql
# MAGIC -- INSERT INTO `31184_cerebro_prd`.`cv0361`.`order_stock`(OrderType, TransactionId, AccountId, Symbol, Price, Quantity, OrderDatetime) 
# MAGIC -- VALUES ('BUY','6191acac-ff55-4a08-8440-4dcf37ed13a6', '23456','T', 21.95, 100, current_timestamp());
# MAGIC -- INSERT INTO `31184_cerebro_prd`.`cv0361`.`order_stock`(OrderType, TransactionId, AccountId, Symbol, Price, Quantity, OrderDatetime) 
# MAGIC -- VALUES ('BUY','56ab9048-0b5d-4523-af80-dc11ce51eb9f', '23456','AAPL', 231.30, 15, current_timestamp());
# MAGIC -- INSERT INTO `31184_cerebro_prd`.`cv0361`.`order_stock`(OrderType, TransactionId, AccountId, Symbol, Price, Quantity, OrderDatetime) 
# MAGIC -- VALUES ('SELL','56ab9048-0b5d-4523-af80-dc11ce51eb9f', '23456','TSLA', 205.75, 10, current_timestamp());
# MAGIC
# MAGIC Select OrderType, Symbol as Stock, Price, Quantity as Shares, Price*Quantity as Total, TransactionId, OrderDatetime
# MAGIC From `31184_cerebro_prd`.`cv0361`.`order_stock`
# MAGIC WHERE AccountId = '23456'
# MAGIC Order By OrderDatetime Desc
# MAGIC limit 3

# COMMAND ----------


