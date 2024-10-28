# Databricks notebook source
import mlflow
import pandas as pd
import json
run_id = "200a31cc109443b18c1915308341fd88" # openAI
model_uri =f"runs:/{run_id}/wallstreet_model"
loaded_model = mlflow.pyfunc.load_model(model_uri)
stock_chat_history = [{"query": "what is AT&T?", "answer": "AT&T Inc. is a leading provider of telecommunications and technology services, offering a wide range of products and services to both individual and business customers. The company has a significant presence in the United States and Mexico, operating under various brand names and providing essential communication services."},{"query": "what is the stock price?", "answer": "Based on the provided historical data, the most recent closing price for AT&T Inc. (symbol: T) is from September 27, 2024, which is **$21.90**. This is the latest available stock price in the data you provided."}]
query = "who is the CEO and how much does the CEO make?"

input_param = {"query":query,"chat_history":stock_chat_history}#,"query_filter":query_filter})
input_df = pd.DataFrame({
    "query": [query],
    "chat_history": [json.dumps(stock_chat_history)],  # Assuming the model expects this format; adjust if necessary
})
predictions = loaded_model.predict(input_df)
print(predictions.get("response"))
print(predictions.get("citations"))

# COMMAND ----------


