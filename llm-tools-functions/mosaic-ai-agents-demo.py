# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Mosaic AI Agent Framework & Agent Evaluation demo
# MAGIC
# MAGIC This tutorial shows you how to build, deploy, and evaluate a RAG application using Mosaic AI Agent Framework ([AWS](https://docs.databricks.com/generative-ai/retrieval-augmented-generation.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/retrieval-augmented-generation)) and Mosaic AI Agent Evaluation ([AWS](https://docs.databricks.com/generative-ai/agent-evaluation/index.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-evaluation/)). In this tutorial, you:
# MAGIC
# MAGIC 1. Build a vector search index using sample data chunks.
# MAGIC 2. Deploy a RAG application built with Agent Framework.
# MAGIC 3. Evaluate the quality of the application with Agent Evaluation and MLflow.
# MAGIC
# MAGIC In this example, you build a RAG chatbot that can answer questions using information from Databricks public documentation ([AWS](https://docs.databricks.com) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/)).
# MAGIC
# MAGIC ## Requirements
# MAGIC
# MAGIC  - This notebook requires a single-user cluster ([AWS](https://docs.databricks.com/en/compute/configure.html#access-modes) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/compute/configure#access-mode)) running on Databricks Runtime 14.3 and above.
# MAGIC  - Agent Framework and Agent Evaluation are only available on Amazon Web Services and Azure cloud platforms.
# MAGIC
# MAGIC ## Databricks features used in this demo:
# MAGIC - **Agent Framework** ([AWS](https://docs.databricks.com/generative-ai/retrieval-augmented-generation.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/retrieval-augmented-generation)) - An SDK used to quickly and safely build high-quality RAG applications.
# MAGIC - **Agent Evaluation** ([AWS](https://docs.databricks.com/generative-ai/agent-evaluation/index.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-evaluation/)) - AI-assisted tools that help evaluate if outputs are high-quality. Include an intuitive UI-based review app to get feedback from human stakeholders.
# MAGIC - **Mosaic AI Model Serving** ([AWS](https://docs.databricks.com/generative-ai/deploy-agent.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/deploy-agent)) Hosts the application's logic as a production-ready, scalable REST API.
# MAGIC - **MLflow** ([AWS](https://docs.databricks.com/mlflow/index.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/mlflow/)) Tracks and manages the application lifecycle, including evaluation results and application code/config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install dependencies
# MAGIC
# MAGIC Install the necessary dependencies and specify versions for compatibility.

# COMMAND ----------

# MAGIC %pip uninstall -y mlflow mlflow-skinny
# MAGIC %pip install -U -qqqq databricks-agents mlflow mlflow-skinny databricks-vectorsearch databricks-sdk langchain==0.2.1 langchain_core==0.2.5 langchain_community==0.2.4 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup: Load the necessary data and code from the Databricks Cookbook repo
# MAGIC
# MAGIC Clone the Generative AI cookbook repo from `https://github.com/databricks/genai-cookbook` into a folder `genai-cookbook` in the same folder as this notebook using a Git Folder ([AWS](https://docs.databricks.com/en/repos/index.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/repos/)).  
# MAGIC
# MAGIC Alternatively, you can manually clone the Git repo `https://github.com/databricks/genai-cookbook` to a folder `genai-cookbook`.

# COMMAND ----------

import os
from databricks.sdk.core import DatabricksError
from databricks.sdk import WorkspaceClient

CURRENT_FOLDER = os.getcwd()
QUICK_START_REPO_URL = "https://github.com/databricks/genai-cookbook.git"
QUICK_START_REPO_SAVE_FOLDER = "genai-cookbook"

if os.path.isdir(QUICK_START_REPO_SAVE_FOLDER):
    raise Exception(
        f"{QUICK_START_REPO_SAVE_FOLDER} folder already exists, please change the variable QUICK_START_REPO_SAVE_FOLDER to be a non-existant path."
    )

# Clone the repo
w = WorkspaceClient()
try:
    w.repos.create(
        url=QUICK_START_REPO_URL, provider="github", path=f"{CURRENT_FOLDER}/{QUICK_START_REPO_SAVE_FOLDER}"
    )
    print(f"Cloned sample code repo to: {QUICK_START_REPO_SAVE_FOLDER}")
except DatabricksError as e:
    if e.error_code == "RESOURCE_ALREADY_EXISTS":
        print("Repo already exists. Skipping creation")
    else:
        raise Exception(
            f"Failed to clone the quick start code.  You can manually import this by creating a Git folder from the contents of {QUICK_START_REPO_URL} in the {QUICK_START_REPO_SAVE_FOLDER} folder in your workspace and then re-running this Notebook."
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalog and application setup
# MAGIC
# MAGIC Set the catalog and schema where the following resources will be registered:
# MAGIC
# MAGIC - `UC_CATALOG` and `UC_SCHEMA`: Unity Catalog ([AWS](https://docs.databricks.com/en/data-governance/unity-catalog/create-catalogs.html#create-a-catalog) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/catalogs/)) and a Schema where the output Delta tables and Vector Search indexes are stored
# MAGIC - `UC_MODEL_NAME`: Unity Catalog location to log and store the chain's model
# MAGIC - `VECTOR_SEARCH_ENDPOINT`: Vector Search Endpoint ([AWS](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-a-vector-search-endpoint) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/create-query-vector-search)) to host the vector index
# MAGIC
# MAGIC You must have `USE CATALOG` privilege on the catalog, and `CREATE MODEL` and `USE SCHEMA` privileges on the schema. 
# MAGIC
# MAGIC Change the catalog and schema here if necessary. Any missing resources will be created in the next step.

# COMMAND ----------

# Use the current user name to create any necessary resources
w = WorkspaceClient()
user_name = w.current_user.me().user_name.split("@")[0].replace(".", "")

# UC Catalog and Schema where outputs tables/indexs are saved
# If this catalog/schema does not exist, you need create catalog/schema permissions.
UC_CATALOG = f'{user_name}_catalog'
UC_SCHEMA = f'agent_demo'

# UC Model name where tr chain is logged
UC_MODEL_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.doc_bot"

# Vector Search endpoint where the index is loaded
# If this does not exist, it will be created
VECTOR_SEARCH_ENDPOINT = f'{user_name}_vector_search'

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Create the UC Catalog, UC Schema, and Vector Search endpoint
# MAGIC
# MAGIC Check if the UC resources exist. Create the resources if they don't exist.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import EndpointStatusState, EndpointType
from databricks.sdk.service.serving import EndpointCoreConfigInput, EndpointStateReady
from databricks.sdk.errors import ResourceDoesNotExist, NotFound, PermissionDenied
import os
w = WorkspaceClient()

# Create a UC Catalog if it does not exist, otherwise, raise an exception
try:
    _ = w.catalogs.get(UC_CATALOG)
    print(f"PASS: UC catalog `{UC_CATALOG}` exists")
except NotFound as e:
    print(f"`{UC_CATALOG}` does not exist, trying to create...")
    try:
        _ = w.catalogs.create(name=UC_CATALOG)
    except PermissionDenied as e:
        print(f"FAIL: `{UC_CATALOG}` does not exist, and no permissions to create.  Please provide an existing UC Catalog.")
        raise ValueError(f"Unity Catalog `{UC_CATALOG}` does not exist.")
        
# Create UC Schema if it does not exist, otherwise, raise an exception
try:
    _ = w.schemas.get(full_name=f"{UC_CATALOG}.{UC_SCHEMA}")
    print(f"PASS: UC schema `{UC_CATALOG}.{UC_SCHEMA}` exists")
except NotFound as e:
    print(f"`{UC_CATALOG}.{UC_SCHEMA}` does not exist, trying to create...")
    try:
        _ = w.schemas.create(name=UC_SCHEMA, catalog_name=UC_CATALOG)
        print(f"PASS: UC schema `{UC_CATALOG}.{UC_SCHEMA}` created")
    except PermissionDenied as e:
        print(f"FAIL: `{UC_CATALOG}.{UC_SCHEMA}` does not exist, and no permissions to create.  Please provide an existing UC Schema.")
        raise ValueError("Unity Catalog Schema `{UC_CATALOG}.{UC_SCHEMA}` does not exist.")

# Create the Vector Search endpoint if it does not exist
vector_search_endpoints = w.vector_search_endpoints.list_endpoints()
if sum([VECTOR_SEARCH_ENDPOINT == ve.name for ve in vector_search_endpoints]) == 0:
    print(f"Please wait, creating Vector Search endpoint `{VECTOR_SEARCH_ENDPOINT}`.  This can take up to 20 minutes...")
    w.vector_search_endpoints.create_endpoint_and_wait(VECTOR_SEARCH_ENDPOINT, endpoint_type=EndpointType.STANDARD)

# Make sure the Vector Search endpoint is online and ready.
w.vector_search_endpoints.wait_get_endpoint_vector_search_endpoint_online(VECTOR_SEARCH_ENDPOINT)

print(f"PASS: Vector Search endpoint `{VECTOR_SEARCH_ENDPOINT}` exists")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Build and deploy the application
# MAGIC
# MAGIC The following is a high-level overview of the architecture you will deploy:
# MAGIC
# MAGIC 1. Data preparation
# MAGIC     - Copy the sample data to Delta table.
# MAGIC     - Create a Vector Search index using the `databricks-gte-large-en` foundation embedding model.
# MAGIC 2. Inferences
# MAGIC     - Configure the chain, register the chain as an MLflow model, and set up trace logging.
# MAGIC     - Register the application in Unity Catalog.
# MAGIC     - Deploy the chain.
# MAGIC
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-basic.png?raw=true" style="width: 800px; margin-left: 10px">

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-basic-prep-3.png?raw=true" style="float: right; margin-left: 10px" width="400px">
# MAGIC
# MAGIC ## Create the Vector Search Index
# MAGIC
# MAGIC Copy the sample data to a Delta table and sync it to a Vector Search index. Use the [gte-large-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5) embedding model hosted on Databricks Foundational Model APIs ([AWS](https://docs.databricks.com/en/machine-learning/foundation-models/index.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-models/)) to create the vector embeddings.

# COMMAND ----------

# UC locations to store the chunked documents and index
CHUNKS_DELTA_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.databricks_docs_chunked"
CHUNKS_VECTOR_INDEX = f"{UC_CATALOG}.{UC_SCHEMA}.databricks_docs_chunked_index"

# COMMAND ----------

from pyspark.sql import SparkSession
from databricks.vector_search.client import VectorSearchClient

# Workspace URL for printing links to the delta table/vector index
workspace_url = SparkSession.getActiveSession().conf.get(
    "spark.databricks.workspaceUrl", None
)

# Vector Search client
vsc = VectorSearchClient(disable_notice=True)

# Load the chunked data to Delta table and enable change-data capture to allow the table to sync to Vector Search
chunked_docs_df = spark.read.parquet(
    f"file:{CURRENT_FOLDER}/{QUICK_START_REPO_SAVE_FOLDER}/quick_start_demo/chunked_databricks_docs.snappy.parquet"
)
chunked_docs_df.write.format("delta").mode("overwrite").saveAsTable(CHUNKS_DELTA_TABLE)
spark.sql(
    f"ALTER TABLE {CHUNKS_DELTA_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)

print(
    f"View Delta Table at: https://{workspace_url}/explore/data/{UC_CATALOG}/{UC_SCHEMA}/{CHUNKS_DELTA_TABLE.split('.')[-1]}"
)

# Embed and sync chunks to a vector index
print(
    f"Embedding docs & creating Vector Search Index, this will take ~5 - 10 minutes.\nView Index Status at: https://{workspace_url}/explore/data/{UC_CATALOG}/{UC_SCHEMA}/{CHUNKS_VECTOR_INDEX.split('.')[-1]}"
)

index = vsc.create_delta_sync_index_and_wait(
    endpoint_name=VECTOR_SEARCH_ENDPOINT,
    index_name=CHUNKS_VECTOR_INDEX,
    primary_key="chunk_id",
    source_table_name=CHUNKS_DELTA_TABLE,
    pipeline_type="TRIGGERED",
    embedding_source_column="chunked_text",
    embedding_model_endpoint_name="databricks-gte-large-en",
)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Deploy to the review application
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-basic-chain-1.png?raw=true" style="float: right" width="500px">
# MAGIC
# MAGIC Now that the Vector Search index is ready, prepare the RAG chain and deploy it to the review application backed by a scalable-production ready REST API on Model serving.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure chain parameters
# MAGIC
# MAGIC Databricks makes it easy to parameterize your chain with MLflow Model Configurations. Later, you can tune your application by adjusting parameters such as the system prompt or retrieval settings.
# MAGIC
# MAGIC This demo keeps configurations to a minimum, but most applications will include many more parameters to tune.

# COMMAND ----------

chain_config = {
    "llm_model_serving_endpoint_name": "databricks-dbrx-instruct",  # The foundation model we want to use
    "vector_search_endpoint_name": VECTOR_SEARCH_ENDPOINT,  # Endoint for Vector Search
    "vector_search_index": f"{CHUNKS_VECTOR_INDEX}",
    "llm_prompt_template": """You are an assistant that answers questions. Use the following pieces of retrieved context to answer the question. Some pieces of context may be irrelevant, in which case you should not use them to form the answer.\n\nContext: {context}""", # LLM Prompt template
}

# Define an input example in the schema required by Agent Framework
input_example = {"messages": [ {"role": "user", "content": "What is Retrieval-augmented Generation?"}]}

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Log the application & view an MLflow trace
# MAGIC
# MAGIC Register the chain as an MLflow model and inspect the MLflow trace to understand what is happening inside the chain.
# MAGIC
# MAGIC <br/>
# MAGIC <img src="https://ai-cookbook.io/_images/mlflow_trace2.gif" width="80%" style="margin-left: 10px">
# MAGIC

# COMMAND ----------

import mlflow

# Log the model to MLflow
with mlflow.start_run(run_name="databricks-docs-bot"):
    logged_chain_info = mlflow.langchain.log_model(
        lc_model=os.path.join(
            os.getcwd(),
            f"{QUICK_START_REPO_SAVE_FOLDER}/quick_start_demo/sample_rag_chain",
        ),  # Chain code file from the quick start repo
        model_config=chain_config,  # Chain configuration set above
        artifact_path="chain",  # Required by MLflow
        input_example=input_example,  # Save the chain's input schema.  MLflow will execute the chain before logging and capturing its output schema.
    )

# Test the chain locally to see the MLflow Trace
chain = mlflow.langchain.load_model(logged_chain_info.model_uri)
chain.invoke(input_example)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Deploy the application
# MAGIC
# MAGIC To deploy to the application:
# MAGIC
# MAGIC 1. Register the application in Unity Catalog.
# MAGIC 2. Use Agent Framework to deploy to the Agent Evaluation review app.
# MAGIC
# MAGIC In addition to the review app, a scalable, production-ready Model Serving endpoint is deployed.

# COMMAND ----------

from databricks import agents
import time
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate

# Use Unity Catalog to log the chain
mlflow.set_registry_uri('databricks-uc')

# Register the chain to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, name=UC_MODEL_NAME)

# Deploy to enable the Review APP and create an API endpoint
deployment_info = agents.deploy(model_name=UC_MODEL_NAME, model_version=uc_registered_model_info.version)

# Wait for the Review App to be ready
print("\nWaiting for endpoint to deploy.  This can take 10 - 20 minutes.", end="")
while w.serving_endpoints.get(deployment_info.endpoint_name).state.ready == EndpointStateReady.NOT_READY or w.serving_endpoints.get(deployment_info.endpoint_name).state.config_update == EndpointStateConfigUpdate.IN_PROGRESS:
    print(".", end="")
    time.sleep(30)

# COMMAND ----------

# MAGIC %md
# MAGIC # Evaluate the application
# MAGIC
# MAGIC Once the application is deployed, you can evaluate its quality.
# MAGIC
# MAGIC - Human reviewers can use the review app to interact with the application and provide feedback on responses.
# MAGIC - Chain metrics provide quality metrics such as latency and token use.
# MAGIC - LLM judges use external large language models to analyze the output of your application and judge the quality of retrieved chunks and generated responses.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Get feedback from human reviewers
# MAGIC
# MAGIC Have domain experts test the bot by chatting with it and providing correct answers when the bot doesn't respond properly. This is a critical step to build or improve your evaluation dataset.
# MAGIC
# MAGIC Your evaluation dataset forms the basis of your development workflow to improve quality: identify the root causes of quality issues and then objectively measure the impact of your fixes.
# MAGIC
# MAGIC The application automatically captures all stakeholder questions, stakeholder feedback, bot responses, and MLflow traces into Delta tables. 
# MAGIC
# MAGIC **Your domain experts do NOT need to have Databricks workspace access** - you can assign permissions to any user in your SSO if you have enabled SCIM ([AWS](https://docs.databricks.com/en/admin/users-groups/scim/index.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/admin/users-groups/scim/)).
# MAGIC
# MAGIC <br/>
# MAGIC
# MAGIC <img src="https://ai-cookbook.io/_images/review_app2.gif" style="float: left;  margin-left: 10px" width="80%">

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Evaluate to get chain metrics
# MAGIC
# MAGIC Use Agent Evaluation's specialized AI evaluators to assess chain performance without the need for human reviewers. Agent Evaluation is integrated into `mlflow.evaluate(...)` all you need to do is pass `model_type="databricks-agent"`. 
# MAGIC
# MAGIC There are three types of evaluation metrics:
# MAGIC
# MAGIC - **Ground truth based:** Assess performance based on known correct answers. Compare the RAG application’s retrieved documents or generated outputs to the ground truth documents and answers recorded in the evaluation set.
# MAGIC
# MAGIC - **LLM judge-based:** A separate LLM acts as a judge to evaluate the RAG application’s retrieval and response quality. This approach automates evaluation across numerous dimensions.
# MAGIC
# MAGIC - **Trace metrics:** Metrics computed using the agent trace help determine quantitative metrics like agent cost and latency.
# MAGIC
# MAGIC <br>
# MAGIC <img src="https://ai-cookbook.io/_images/mlflow-eval-agent.gif" style="float: left;  margin-left: 10px" width="80%">

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the evaluation set
# MAGIC
# MAGIC This demo uses a toy 4-question evaluation dataset. To learn more about evaluation best practices, see best practices ([AWS](https://docs.databricks.com/ai-cookbook/implementation/step-3-evaluation-set.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/tutorials/ai-cookbook/implementation/step-3-evaluation-set))
# MAGIC .

# COMMAND ----------

import pandas as pd

sample_eval_set = [
    {
        "request_id": "5482",
        "request": "What happens if I try to access an index that is out of bounds in an array using the [ ] operator in Databricks SQL when spark.sql.ansi.enabled is set to false?",
        "expected_response": "If you try to access an index that is out of bounds in an array using the [ ] operator in Databricks SQL when spark.sql.ansi.enabled is set to false, Databricks will return NULL instead of raising an error.",
    },
    {
        "request_id": "2112",
        "request": "Why is a long-running stage in my Spark job only showing one task, and how can I resolve this issue?",
        "expected_response": "A long-running stage with one task in a Spark job could be due to several reasons such as:\n\n1. Expensive User Defined Functions (UDFs) on small data\n2. Window function without a PARTITION BY statement\n3. Reading from an unsplittable file type like gzip\n4. Setting the multiLine option when reading a JSON or CSV file\n5. Schema inference of a large file\n6. Use of repartition(1) or coalesce(1)\n\nTo resolve this issue, you can:\n\n1. Optimize your UDFs or replace them with built-in functions if possible.\n2. Ensure that you have a proper PARTITION BY statement in your window functions.\n3. Avoid using unsplittable file types like gzip. Instead, use splittable file types like snappy or lz4.\n4. Avoid setting the multiLine option when reading JSON or CSV files.\n5. Perform schema inference on a small sample of your data and then apply it to the entire dataset.\n6. Avoid using repartition(1) or coalesce(1) unless necessary.\n\nBy implementing these changes, you should be able to resolve the issue of a long-running stage with only one task in your Spark job.",
    },
    {
        "request_id": "5054",
        "request": "How can I represent 4-byte single-precision floating point numbers in Databricks SQL and what are their limits?",
        "expected_response": "4-byte single-precision floating point numbers can be represented in Databricks SQL using the `FLOAT` or `REAL` syntax. The range of numbers that can be represented is from -3.402E+38 to +3.402E+38, including negative infinity, positive infinity, and NaN (not a number). Here are some examples of how to represent these numbers:\n\n* `+1F` represents 1.0\n* `5E10F` represents 5E10\n* `5.3E10F` represents 5.3E10\n* `-.1F` represents -0.1\n* `2.F` represents 2.0\n* `-5555555555555555.1F` represents -5.5555558E15\n* `CAST(6.1 AS FLOAT)` represents 6.1\n\nNote that `FLOAT` is a base-2 numeric type, so the representation of base-10 literals may not be exact. If you need to accurately represent fractional or large base-10 numbers, consider using the `DECIMAL` type instead.",
    },
    {
        "request_id": "2003",
        "request": "How can I identify the reason for failing executors in my Databricks workspace, and what steps can I take to resolve memory issues?",
        "expected_response": "1. Identify failing executors: In your Databricks workspace, navigate to the compute's Event log to check for any explanations regarding executor failures. Look for messages indicating spot instance losses or cluster resizing due to autoscaling. If using spot instances, refer to 'Losing spot instances' documentation. For autoscaling, refer to 'Learn more about cluster resizing' documentation.\n\n2. Check executor logs: If no information is found in the event log, go to the Spark UI and click the Executors tab. Here, you can access logs from failed executors to investigate further.\n\n3. Identify memory issues: If the above steps do not provide a clear reason for failing executors, it is likely a memory issue. To dig into memory issues, refer to the 'Spark memory issues' documentation.\n\n4. Resolve memory issues: To resolve memory issues, consider the following steps:\n\n   a. Increase executor memory: Allocate more memory to executors by adjusting the 'spark.executor.memory' property in your Spark configuration.\n\n   b. Increase driver memory: Allocate more memory to the driver by adjusting the 'spark.driver.memory' property in your Spark configuration.\n\n   c. Use off-heap memory: Enable off-heap memory by setting the 'spark.memory.offHeap.enabled' property to 'true' and allocating off-heap memory using the 'spark.memory.offHeap.size' property.\n\n   d. Optimize data processing: Review your data processing workflows and optimize them for memory efficiency. This may include reducing data shuffling, using broadcast variables, or caching data strategically.\n\n   e. Monitor memory usage: Monitor memory usage in your Databricks workspace to identify potential memory leaks or inefficient memory utilization. Use tools like the Spark UI, Ganglia, or Grafana to monitor memory usage.",
    },
]

eval_df = pd.DataFrame(sample_eval_set)
display(eval_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Run evaluation

# COMMAND ----------

with mlflow.start_run(run_id=logged_chain_info.run_id):
    # Evaluate
    eval_results = mlflow.evaluate(
        data=eval_df,  # Your evaluation set
        model=logged_chain_info.model_uri,  # previously logged model
        model_type="databricks-agent",  # activate Mosaic AI Agent Evaluation
    )

# COMMAND ----------

# MAGIC %md
# MAGIC # Next steps
# MAGIC
# MAGIC
# MAGIC ## Code-based quickstarts
# MAGIC
# MAGIC | Time required | Outcome | Link |
# MAGIC |------ | ---- | ---- |
# MAGIC | 🕧🕧 <br/>30 minutes | Comprehensive quality/cost/latency evaluation of your proof of concept app | - Evaluate your proof of concept ([AWS](https://docs.databricks.com/ai-cookbook/implementation/step-4-evaluate-quality.html) \| [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/tutorials/ai-cookbook/implementation/step-4-evaluate-quality)) <br/> - Identify the root causes of quality issues ([AWS](https://docs.databricks.com/en/generative-ai/tutorials/ai-cookbook/implementation/step-5-root-cause-analysis.html) \| [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/tutorials/ai-cookbook/implementation/step-5-root-cause-analysis)) |
# MAGIC
# MAGIC ## Browse the code samples
# MAGIC
# MAGIC Open the `./genai-cookbook/rag_app_sample_code` folder this notebook synced to your Workspace.  Documentation here ([AWS](https://docs.databricks.com/en/generative-ai/tutorials/ai-cookbook/implementation/step-6-improve-quality.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/tutorials/ai-cookbook/implementation/step-6-improve-quality)).
# MAGIC
# MAGIC ## Read the Generative AI Cookbook ([AWS](https://docs.databricks.com/en/generative-ai/tutorials/ai-cookbook/introduction.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/tutorials/ai-cookbook/introduction))
# MAGIC
# MAGIC The Databricks Generative AI Cookbook is a definitive how-to guide for building *high-quality* generative AI applications. *High-quality* applications are applications that:
# MAGIC 1. **Accurate:** provide correct responses
# MAGIC 2. **Safe:** do not deliver harmful or insecure responses
# MAGIC 3. **Governed:** respect data permissions and access controls and track lineage
