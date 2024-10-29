# Databricks notebook source
# MAGIC %md
# MAGIC # Stock Exection Agent Based on Mosaic AI Agent Framework
# MAGIC * Get Realtime Stock Quote - How is Apple performing?
# MAGIC * Buy some shares of stock - Grab me 100 shares of AT&T.
# MAGIC * Sell some shares of stock - Sell 15 shares of Testla.
# MAGIC * Get recent transaction history - What are my latest transactions?
# MAGIC
# MAGIC ### References:
# MAGIC * https://docs.databricks.com/en/generative-ai/tutorials/agent-framework-notebook.html
# MAGIC * https://notebooks.databricks.com/demos/llm-tools-functions/index.html#
# MAGIC * https://learn.microsoft.com/en-us/azure/databricks/generative-ai/create-log-agent

# COMMAND ----------

# MAGIC %pip install -U databricks-sdk==0.23.0 langchain-community==0.2.10 langchain-openai==0.1.19 mlflow==2.14.3

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

catalog = "31184_cerebro_prd"
dbName = db = "cv0361"

# COMMAND ----------

print(f"USE CATALOG `{catalog}`")
spark.sql(f"USE CATALOG `{catalog}`")

print(f"using catalog.database `{catalog}`.`{db}`")
spark.sql(f"""USE `{catalog}`.`{db}`""")    


# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Function - Get Realtime Stock Quote

# COMMAND ----------

# %sql
# drop function 31184_cerebro_prd.cv0361.get_realtime_stock

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION get_realtime_stock(stock_symbol STRING)
# MAGIC RETURNS STRING
# MAGIC COMMENT 'This function retrieves realtime stock data from Yahoo Finance. This function take Stock Symbol as input parameter.'
# MAGIC LANGUAGE SQL
# MAGIC     RETURN
# MAGIC     Select concat('Stock Price: ', `Last Sale`, '; Net Change: ', `Net Change`, '; Percent Change: ', `% Change`) as RealtimeQuote
# MAGIC     From stock
# MAGIC     Where Symbol = stock_symbol
# MAGIC     limit 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- let's test our function:
# MAGIC SELECT get_realtime_stock('TSLA');

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Function - Stock Ordering (Buy/Sell)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION execute_stock_order(
# MAGIC   AccountId string,
# MAGIC   OrderType string,
# MAGIC   Symbol string,
# MAGIC   Quantity int
# MAGIC )
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON
# MAGIC COMMENT 'This function buy or sell stock. This function take Account Id, Order Type, Stock Symbol, Quantity as input parameters.'
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
# MAGIC SELECT execute_stock_order('23456', 'BUY', 'T', 50) as TransactionId;
# MAGIC -- SELECT execute_stock_order('23456', 'SELL', 'T', 100) as TransactionId;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Function - Get Recent Stock Transactions

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
# MAGIC COMMENT 'Returns a list of stock execution orders history for a customer given the Account ID.'
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

# MAGIC %md
# MAGIC ## Working with Agent

# COMMAND ----------

# %pip install -U databricks-sdk==0.23.0 langchain-community==0.2.10 langchain-openai==0.1.19 mlflow==2.14.3

# COMMAND ----------

# dbutils.library.restartPython()

# COMMAND ----------

import mlflow
mlflow.langchain.autolog(disable=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get a List of Tools to use

# COMMAND ----------

from langchain_community.tools.databricks import UCFunctionToolkit
import pandas as pd

from databricks.sdk import WorkspaceClient

def get_shared_warehouse(name=None):
    w = WorkspaceClient()
    warehouses = w.warehouses.list()
    for wh in warehouses:
        if wh.name == name:
            return wh
    for wh in warehouses:
        if wh.name.lower() == "shared endpoint":
            return wh
    for wh in warehouses:
        if wh.name.lower() == "dbdemos-shared-endpoint":
            return wh
    #Try to fallback to an existing shared endpoint.
    for wh in warehouses:
        if "dbdemos" in wh.name.lower():
            return wh
    for wh in warehouses:
        if "shared" in wh.name.lower():
            return wh
    for wh in warehouses:
        if wh.num_clusters > 0:
            return wh       
    raise Exception("Couldn't find any Warehouse to use. Please create a wh first to run the demo and add the id here")


def display_tools(tools):
    display(pd.DataFrame([{k: str(v) for k, v in vars(tool).items()} for tool in tools]))

wh = get_shared_warehouse(name = None) #Get the first shared wh we can. See _resources/01-init for details
print(f'This demo will be using the SQL Warehouse: {wh.name} to execute the functions')


# COMMAND ----------

def get_tools():
    return (
        UCFunctionToolkit(warehouse_id=wh.id)
        # Include functions as tools using their qualified names.
        # You can use "{catalog_name}.{schema_name}.*" to get all functions in a schema.
        .include(f"{catalog}.{db}.*")
        .get_tools())

display_tools(get_tools()) #display in a table the tools - see _resource/00-init for details

# COMMAND ----------

# %sql
# drop function recommend_outfit_description

# COMMAND ----------

# MAGIC %md
# MAGIC ### Formulate LLM object

# COMMAND ----------

from langchain_openai import ChatOpenAI
from databricks.sdk import WorkspaceClient

# Note: langchain_community.chat_models.ChatDatabricks doesn't support create_tool_calling_agent yet - it'll soon be availableK. Let's use ChatOpenAI for now
llm = ChatOpenAI(
  base_url=f"{WorkspaceClient().config.host}/serving-endpoints/",
  api_key=dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get(),
  model="databricks-meta-llama-3-70b-instruct"
)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Formulate Prompt

# COMMAND ----------

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatDatabricks

def get_prompt(history = [], prompt = None):
    if not prompt:
            prompt = """You are a stockbroker. Your task is to perform users commands by invoking the following tools:
    - Use the execute_stock_order to buy or sell stock. This function take Account Id, Order Type, Stock Symbol, Quantity as input parameters.
    - Use get_stock_orders to get a list of stock execution orders history for a customer given the Account ID. 
    - Use get_realtime_stock to retrieves realtime stock data from Yahoo Finance. This function take Stock Symbol as input parameter.

    Make sure to use the appropriate tool for each step and provide a coherent response to the user. Don't mention tools to your users. Only answer what the user is asking for. If the question isn't related to the tools or style/clothe, say you're sorry but can't answer"""
    return ChatPromptTemplate.from_messages([
            ("system", prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
    ])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Agent

# COMMAND ----------

from langchain.agents import AgentExecutor, create_tool_calling_agent

prompt = get_prompt()
tools = get_tools()
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Testdrive the Agent

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get Realtime Stock Info and Selling Stock

# COMMAND ----------

agent_executor.invoke({"input": "what is tesla stock price?"})

# COMMAND ----------

agent_executor.invoke({"input": "sell 10 shares of tesla."})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get Realtime Stock Performance and Buy Stock

# COMMAND ----------

agent_executor.invoke({"input": "how is AT&T performing?"})

# COMMAND ----------

agent_executor.invoke({"input": "I want to buy 100 shares of AT&T."})

# COMMAND ----------

agent_executor.invoke({"input": "can you grab me 15 shares of Apple?"})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checking Stock Order Transaction History

# COMMAND ----------

agent_executor.invoke({"input": "what are my stock orders for account 23456?"})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploying our Agent Executor as Model Serving Endpoint
# MAGIC
# MAGIC We're now ready to package our chain within MLFLow, and deploy it as a Model Serving Endpoint, leveraging Databricks Agent Framework and its review application.

# COMMAND ----------

import os

# COMMAND ----------

# # TODO: write this as a separate file if you want to deploy it properly
# from langchain.schema.runnable import RunnableLambda
# from langchain_core.output_parsers import StrOutputParser

# # Function to extract the user's query
# def extract_user_query_string(chat_messages_array):
#     return chat_messages_array[-1]["content"]

# # Wrapping the agent_executor invocation
# def agent_executor_wrapper(input_data):
#     result = agent_executor.invoke({"input": input_data})
#     return result["output"]

# # Create the chain using the | operator with StrOutputParser
# chain = (
#     RunnableLambda(lambda data: extract_user_query_string(data["messages"]))  # Extract the user query
#     | RunnableLambda(agent_executor_wrapper)  # Pass the query to the agent executor
#     | StrOutputParser()  # Optionally parse the output to ensure it's a clean string
# )

# COMMAND ----------

# Example input data
input_data = {
    "messages": [
        {"content": "what are my stock orders for account 23456?"}
    ]
}

# # Run the chain
# answer = chain.invoke(input_data)
# displayHTML(answer.replace('\n', '<br>'))

# COMMAND ----------

# For this first basic demo, we'll keep the configuration as a minimum. In real app, you can make all your RAG as a param (such as your prompt template to easily test different prompts!)
chain_config = {
    "llm_model": "databricks-meta-llama-3-70b-instruct",
    "warehouse_id": wh.id,
    "api_key": dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get(),
    "catalog": catalog,
    "db": dbName
} 


# COMMAND ----------

os.path.join(os.getcwd(), 'chain_notebook')

# COMMAND ----------

# Log the model to MLflow
with mlflow.start_run(run_name="stock_agent_bot"):
    logged_chain_info = mlflow.langchain.log_model(
            lc_model=os.path.join(os.getcwd(), 'chain_notebook'), # Chain code file e.g., /path/to/the/chain.py 
            model_config=chain_config, # Chain configuration 
            artifact_path="chain", # Required by MLflow, the chain's code/config are saved in this directory
            input_example=input_data,
            example_no_conversion=True,  # Required by MLflow to use the input_example as the chain's schema
            # signature=False
        )



# COMMAND ----------

print(logged_chain_info.model_uri)

# Test the chain locally to see the MLflow Trace
chain = mlflow.langchain.load_model(logged_chain_info.model_uri)


# COMMAND ----------

input_data = {
    "messages": [
        {"content": "what are my stock orders for account 23456?"}
    ]
}
chain.invoke(input_data)

# COMMAND ----------

input_data = {
    "messages": [
        {"content": "stock price for pfizer?"}
    ]
}

chain.invoke(input_data)

# COMMAND ----------

input_data = {
    "messages": [
        {"content": "Dispose 100 shares of Intel."}
    ]
}
chain.invoke(input_data)

# COMMAND ----------

input_data = {
    "messages": [
        {"content": "can you grab me 15 shares of Apple?"}
    ]
}
chain.invoke(input_data)

# COMMAND ----------

# Use the current user name to create any necessary resources
w = WorkspaceClient()
user_name = w.current_user.me().user_name.split("@")[0].replace(".", "")

# UC Catalog and Schema where outputs tables/indexs are saved
# If this catalog/schema does not exist, you need create catalog/schema permissions.
UC_CATALOG = f'31184_cerebro_prd'
UC_SCHEMA = f'cv0361'

# UC Model name where tr chain is logged
UC_MODEL_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.stock_agent_model"

print("UC_MODEL_NAME:", UC_MODEL_NAME)


# COMMAND ----------

import time
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate

# Use Unity Catalog to log the chain
mlflow.set_registry_uri("databricks-uc")

# Register the chain to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, name=UC_MODEL_NAME)



# COMMAND ----------

# from databricks import agents

# # Deploy to enable the Review APP and create an API endpoint
# deployment_info = agents.deploy(model_name=UC_MODEL_NAME, model_version=uc_registered_model_info.version)

# COMMAND ----------

# # Wait for the Review App to be ready
# print("\nWaiting for endpoint to deploy.  This can take 10 - 20 minutes.", end="")

# while w.serving_endpoints.get(deployment_info.endpoint_name).state.ready == EndpointStateReady.NOT_READY or w.serving_endpoints.get(deployment_info.endpoint_name).state.config_update == EndpointStateConfigUpdate.IN_PROGRESS:
#     print(".", end="")
#     time.sleep(30)

# COMMAND ----------


