# Databricks notebook source
# MAGIC %md 
# MAGIC ## Configuration file
# MAGIC
# MAGIC Please change your catalog and schema here to run the demo on a different catalog.
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=2754134964726624&notebook=%2Fconfig&demo_name=llm-tools-functions&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fllm-tools-functions%2Fconfig&version=1">

# COMMAND ----------

VECTOR_SEARCH_ENDPOINT_NAME="dbdemos_vs_endpoint"

catalog = "31184_cerebro_prd"
dbName = db = "cv0361"
volume_name = "dbdemos_agent_volume"
