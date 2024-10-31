# Databricks notebook source
# MAGIC %md
# MAGIC # LLM RAG Evaluation

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import mlflow
from mlflow.metrics.genai import EvaluationExample, faithfulness, relevance
import json
import os

# COMMAND ----------

# use databricks foundation model
from mlflow.deployments import set_deployments_target
set_deployments_target("databricks")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the RAG system using `mlflow.evaluate()`

# COMMAND ----------

# run_id = "09af3d60663143b8ba04c7dbdf671606" #databricks-meta-llama-3-1-405b-instruct
# run_id = "f8c40e79a1ca4245abc1f9697801de87" # openAI
run_id = "d730c421a4084658991bf50a63169f3c" # databricks-llama-2-70b-chat
# run_id = "bc4cc2ff25b7455fa262fdabe5c932ba" # databricks-meta-llama-3-1-70b-instruct

# COMMAND ----------

model_uri =f"runs:/{run_id}/wallstreet_model"
def model(input_df):
    answer = []
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    for index, row in input_df.iterrows():
        stock_chat_history = row["chat_history"]
        query = row["query"]
        input_df = pd.DataFrame({
        "query": [query],
        "chat_history": stock_chat_history,  # Assuming the model expects this format; adjust if necessary
})
        answer.append(loaded_model.predict(input_df))

    return answer

# COMMAND ----------

# MAGIC %md
# MAGIC Load the eval dataset

# COMMAND ----------

eval_df = pd.read_csv("/dbfs/mnt/regression_testing1/hackathon_files/eval_df_v3.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC Create a faithfulness metric

# COMMAND ----------

from mlflow.metrics.genai import EvaluationExample, faithfulness

# Create a good and bad example for faithfulness in the context of this problem
faithfulness_examples = [
    EvaluationExample(
        input="what is the date of the maximum dividend in AT&T ?",
        output="The data of maximun date is April 8, 2021, with a dividend of 0.52",
        score=5,
        justification="The output provides a date of the maximum dividend in AT&T that is provided in the context.",
        grading_context=
            "['Date: 1984-03-26 00:00:00-05:00\nDividends: 0.116667\nStock Splits: 0.0\nFILENAME: ACTIONS', 'Date: 1985-01-04 00:00:00-05:00\nDividends: 0.116667\nStock Splits: 0.0\nFILENAME: ACTIONS', 'Date: 1986-04-04 00:00:00-05:00\nDividends: 0.133333\nStock Splits: 0.0\nFILENAME: ACTIONS', 'Date: 1986-07-03 00:00:00-04:00\nDividends: 0.133333\nStock Splits: 0.0\nFILENAME: ACTIONS', 'Date: 1984-06-25 00:00:00-04:00\nDividends: 0.116667\nStock Splits: 0.0\nFILENAME: ACTIONS']"
    )
]

faithfulness_metric = faithfulness(
    model="endpoints:/databricks-llama-2-70b-chat",examples=faithfulness_examples
)
print(faithfulness_metric)

# COMMAND ----------

relevance_metric = relevance(model="endpoints:/openai")
print(relevance_metric)

# COMMAND ----------

#  **col_mapping**: A dictionary mapping column names in the input dataset or output
#           predictions to column names used when invoking the evaluation functions.
results = mlflow.evaluate(
    model,
    df,
    model_type="question-answering",
    predictions="response",
    evaluators = "default",
    extra_metrics=[faithfulness_metric, relevance_metric, mlflow.metrics.latency()],
    evaluator_config={
        "col_mapping": {
            "inputs": "query",
            "context": "response",
        }
    },
)
print(results.metrics)

# COMMAND ----------

results.tables["eval_results_table"]
