# Databricks notebook source
# MAGIC %pip install -U databricks-sdk==0.23.0 langchain-community==0.2.10 langchain-openai==0.1.19 mlflow==2.14.3

# COMMAND ----------

print("Installed libraries.")

# COMMAND ----------

from langchain_community.tools.databricks import UCFunctionToolkit
import pandas as pd
from databricks.sdk import WorkspaceClient
import mlflow

# Load the chain's configuration
model_config = mlflow.models.ModelConfig(development_config="rag_chain_config.yaml")

warehouse_id = model_config.get("warehouse_id")
catalog = model_config.get("catalog")
db = model_config.get("db")

print(f'SQL Warehouse: {warehouse_id} to execute the functions')

def display_tools(tools):
    display(pd.DataFrame([{k: str(v) for k, v in vars(tool).items()} for tool in tools]))

def get_tools():
    return (
        UCFunctionToolkit(warehouse_id=warehouse_id)
        # Include functions as tools using their qualified names.
        # You can use "{catalog_name}.{schema_name}.*" to get all functions in a schema.
        .include(f"{catalog}.{db}.*")
        .get_tools())

# display_tools(get_tools()) #display in a table the tools - see _resource/00-init for details

from langchain_openai import ChatOpenAI
from databricks.sdk import WorkspaceClient

# Note: langchain_community.chat_models.ChatDatabricks doesn't support create_tool_calling_agent yet - it'll soon be availableK. Let's use ChatOpenAI for now
llm = ChatOpenAI(
  base_url=f"{WorkspaceClient().config.host}/serving-endpoints/",
  api_key= model_config.get("api_key"),
  model=model_config.get("llm_model")
)

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

from langchain.agents import AgentExecutor, create_tool_calling_agent

prompt = get_prompt()
tools = get_tools()
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

#******************************************************************************************************

# TODO: write this as a separate file if you want to deploy it properly
from langchain.schema.runnable import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Function to extract the user's query
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]

# Wrapping the agent_executor invocation
def agent_executor_wrapper(input_data):
    result = agent_executor.invoke({"input": input_data})
    return result["output"]

# Create the chain using the | operator with StrOutputParser
chain = (
    RunnableLambda(lambda data: extract_user_query_string(data["messages"]))  # Extract the user query
    | RunnableLambda(agent_executor_wrapper)  # Pass the query to the agent executor
    | StrOutputParser()  # Optionally parse the output to ensure it's a clean string
)

mlflow.models.set_model(model=chain)

# COMMAND ----------


