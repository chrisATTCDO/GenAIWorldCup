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

def get_tools():
    return (
        UCFunctionToolkit(warehouse_id=wh.id)
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
  api_key=dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get(),
  model="databricks-meta-llama-3-70b-instruct"
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
import mlflow

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