# Databricks notebook source
# MAGIC %pip install langchain_databricks
# MAGIC %pip install databricks-vectorsearch
# MAGIC %pip install databricks-sdk
# MAGIC %pip install langchain_community
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import ast
import base64
import json
from datetime import datetime as dt
from io import BytesIO
from typing import List, Any, Optional

import mlflow
import pandas as pd
from PIL import Image

from databricks.vector_search.client import VectorSearchClient
from langchain_community.embeddings import DatabricksEmbeddings
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.chat_models import ChatDatabricks
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda

from mlflow.models import infer_signature, set_signature
from mlflow.models.signature import ModelSignature
from mlflow.types import ColSpec, DataType, ParamSchema, ParamSpec, Schema

mlflow.langchain.autolog()

config_path="/dbfs/mnt/regression_testing/hackathon_config/wallstreet_config.json"
CHAT_HISTORY_COUNT = 2


# COMMAND ----------

class ExtensionLoader:
    def __init__(self, ingestion_data: dict):
        """
        Initialize the ExtensionLoader with ingestion data.

        :param ingestion_data: Dictionary containing ingestion configuration.
        """
        self.loader_class = ingestion_data.get("file_loader_class_name")
        self.loader_kwargs = ingestion_data.get("loader_kwargs")
        self.splitter_class = ingestion_data.get("splitter_class_name")
        self.splitter_kwargs = ingestion_data.get("splitter_kwargs")

class Config:
    def __init__(self, file_path: str):
        """
        Initialize the Config with a file path.

        :param file_path: Path to the configuration file.
        """
        if not os.path.exists(file_path):
            raise Exception("Config File Path Not present")
        self.__domain_data = json.load(open(file_path, "r"))
        self.__vector_store = self.__domain_data.get("vector_store", None)
        self.__generator = self.__domain_data.get("generator", None)
        self.__prompt = self.__domain_data.get("prompts", None)
        self.__ingestion_configuration = self.__domain_data.get("ingestion", None)
        self.__extension_configs = self.__load_extension_list()

    def __load_extension_list(self) -> dict:
        """
        Load the extension list from the ingestion configuration.

        :return: Dictionary of extension configurations.
        """
        extension_configs = {}
        if self.__ingestion_configuration:
            for ingestion_data in self.__ingestion_configuration:
                for extension in ingestion_data["extension"]:
                    extension_configs[extension] = ExtensionLoader(ingestion_data=ingestion_data)
        return extension_configs

    def get_loader_for_extension(self, extension: str):
        """
        Get the loader for a specific extension.

        :param extension: Extension name.
        :return: ExtensionLoader object or None.
        """
        return self.__extension_configs.get(extension, None)

    def get_embedding_model(self) -> str:
        """
        Get the embedding model name.

        :return: Embedding model name.
        :raises Exception: If model details are not present.
        """
        if self.__vector_store:
            embedding = self.__vector_store.get("embedding", None)
            if embedding:
                model = embedding.get("model", None)
                if model:
                    return model
        raise Exception("Model details Not Present")

    def get_embedding_model_dimension(self) -> int:
        """
        Get the embedding model dimension.

        :return: Embedding model dimension.
        :raises Exception: If model details are not present.
        """
        if self.__vector_store:
            embedding = self.__vector_store.get("embedding", None)
            if embedding:
                dimension = embedding.get("dimension", None)
                if dimension:
                    return dimension
        raise Exception("Model details Not Present")

    def get_vector_index_schema(self) -> dict:
        """
        Get the vector index schema.

        :return: Vector index schema.
        :raises Exception: If index schema is not present.
        """
        if self.__vector_store:
            index = self.__vector_store.get("index", None)
            if index:
                schema = index.get("schema", None)
                if schema:
                    return schema
        raise Exception("Index Schema Not Present")

    def get_vector_index_primary_key(self) -> str:
        """
        Get the vector index primary key.

        :return: Vector index primary key.
        :raises Exception: If primary key information is not present.
        """
        if self.__vector_store:
            index = self.__vector_store.get("index", None)
            if index:
                primary_key = index.get("primary_key", None)
                if primary_key:
                    return primary_key
        raise Exception("Index Primary key information Not Present")

    def get_vector_index_vector_column(self) -> str:
        """
        Get the vector index embedding vector column.

        :return: Embedding vector column name.
        :raises Exception: If embedding vector column information is not present.
        """
        if self.__vector_store:
            index = self.__vector_store.get("index", None)
            if index:
                embedding_vector_column = index.get("embedding_vector_column", None)
                if embedding_vector_column:
                    return embedding_vector_column
        raise Exception("Index Embedding vector column information not Present.")

    def get_vector_endpoint(self) -> str:
        """
        Get the vector endpoint name.

        :return: Vector endpoint name.
        :raises Exception: If endpoint name details are not present.
        """
        if self.__vector_store:
            endpoint_name = self.__vector_store.get("endpoint_name", None)
            if endpoint_name:
                return endpoint_name
        raise Exception("Endpoint_name details Not Present")

    def get_vector_index(self) -> str:
        """
        Get the vector index name.

        :return: Vector index name.
        :raises Exception: If vector index name information is missing.
        """
        if self.__vector_store:
            index = self.__vector_store.get("index", None)
            if index:
                index_name = index.get("name", None)
                if index_name:
                    return index_name
        raise Exception("Vector Index Name information missing in the config not Present")

    def get_generator_endpoint(self) -> str:
        """
        Get the generator endpoint name.

        :return: Generator endpoint name.
        :raises Exception: If LLM endpoint is not present.
        """
        if self.__generator:
            endpoint_name = self.__generator.get("openai_endpoint")
            if endpoint_name:
                return endpoint_name
        raise Exception("LLM endpoint not present")

    def get_generator_model(self) -> str:
        """
        Get the generator model name.

        :return: Generator model name.
        :raises Exception: If LLM model is not present.
        """
        if self.__generator:
            openai_chat_model = self.__generator.get("openai_chat_model")
            if openai_chat_model:
                return openai_chat_model
        raise Exception("LLM Model not present")

    def get_vector_query_type(self) -> str:
        """
        Get the vector query type.

        :return: Vector query type.
        :raises Exception: If query type is not present.
        """
        if self.__vector_store:
            query_type = self.__vector_store.get("query_type")
            if query_type:
                return query_type
        raise Exception("Query Type not present")

    def get_generator_prompt(self) -> str:
        """
        Get the generator prompt.

        :return: Generator prompt.
        :raises Exception: If generator prompt is not present.
        """
        if self.__prompt:
            generator_prompt = self.__prompt.get("generator_prompt")
            if generator_prompt:
                return generator_prompt
        raise Exception("Generator prompt not present")

    def get_symbol_identifier_prompt(self) -> str:
        """
        Get the symbol identifier prompt.

        :return: Symbol identifier prompt.
        :raises Exception: If symbol identifier prompt is not present.
        """
        if self.__prompt:
            symbol_identifier_prompt = self.__prompt.get("symbol_identifier_prompt")
            if symbol_identifier_prompt:
                return symbol_identifier_prompt
        raise Exception("Symbol Identifier prompt not present")

    def get_symbol_conversation_prompt(self) -> str:
        """
        Get the symbol conversation prompt.

        :return: Symbol conversation prompt.
        :raises Exception: If symbol conversation prompt is not present.
        """
        if self.__prompt:
            symbol_conversation_prompt = self.__prompt.get("symbol_conversation_prompt")
            if symbol_conversation_prompt:
                return symbol_conversation_prompt
        raise Exception("Symbol Conversation prompt not present")

    def get_multiturn_prompt(self) -> str:
        """
        Get the multi-turn prompt.

        :return: Multi-turn prompt.
        :raises Exception: If multi-turn prompt is not present.
        """
        if self.__prompt:
            multiturn_prompt = self.__prompt.get("multi-turn_prompt")
            if multiturn_prompt:
                return multiturn_prompt
        raise Exception("Multi-turn prompt not present")

# COMMAND ----------

import os
import json
import base64
import requests
from io import BytesIO
from typing import Optional, List, Dict
from PIL import Image
from databricks.sdk import WorkspaceClient
import mlflow
import pandas as pd
from datetime import datetime as dt


class AzureGpt4VService:
    """
    A service class to interact with Azure's GPT-4V model for generating image descriptions.

    Attributes:
    ----------
    __llm_endpoint : str
        The endpoint URL for the GPT-4V model.
    __llm_api_key : str
        The API key for authenticating requests to the GPT-4V model.

    Methods:
    -------
    get_image_format(base64_source: str) -> str:
        Determines the format of a base64-encoded image.
    image_description(images: List[str], prompt: str, detail_mode: str, image_urls: List[str] = None, deployment_name: Optional[str] = None, llm_endpoint: Optional[str] = None, llm_api_version: Optional[str] = None) -> List[str]:
        Generates descriptions for a list of images using the GPT-4V model.
    """
    def __init__(self, llm_endpoint: Optional[str] = None, llm_api_key: Optional[str] = None):
        self.__llm_endpoint = llm_endpoint
        self.__llm_api_key = llm_api_key

    def get_image_format(self, base64_source: str) -> str:
        """
        Determines the format of a base64-encoded image.

        :param base64_source: Base64 encoded image string.
        :return: Image format.
        """
        image_stream = BytesIO(base64.b64decode(base64_source))
        image = Image.open(image_stream)
        return image.format

    def image_description(
        self,
        images: List[str],
        prompt: str,
        detail_mode: str,
        image_urls: List[str] = None,
        deployment_name: Optional[str] = None,
        llm_endpoint: Optional[str] = None,
        llm_api_version: Optional[str] = None
    ) -> List[str]:
        """
        Generates descriptions for a list of images using the GPT-4V model.

        :param images: List of image URLs.
        :param prompt: Prompt for the GPT-4V model.
        :param detail_mode: Detail mode for the image description.
        :param image_urls: List of base64 encoded image URLs.
        :param deployment_name: Optional deployment name.
        :param llm_endpoint: Optional LLM endpoint.
        :param llm_api_version: Optional LLM API version.
        :return: List of image descriptions.
        """
        messages = [{"role": "system", "content": prompt}]
        content = []
        for i, image in enumerate(image_urls):
            image_format = self.get_image_format(image).lower()
            if image:
                content.append({"type": "text", "text": f"!()[{image_urls[i][0]}]"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/{image_format};base64,{image}", "detail": detail_mode}})
            messages.append({"role": "user", "content": content})
            payload = json.dumps({"messages": messages, "enhancements": {"ocr": {"enabled": False}, "grounding": {"enabled": True}}, "temperature": 0.1, "max_tokens": 1000})
            headers = {
                            'api-key': self.__llm_api_key,
                            'Content-Type': 'application/json'
                        }
            response = requests.post(self.__llm_endpoint, headers=headers, data=payload)
            response = json.loads(response.text)
            page_content = response['choices'][0]['message']['content']
        return page_content
    

class Chat:
    def __init__(self):
        self.__vs_client = VectorSearchClient(disable_notice=True)
        self.__deploy_client = mlflow.deployments.get_deploy_client("databricks")
        self.__config = Config(config_path)
        self.__index = self.__vs_client.get_index(
            index_name=self.__config.get_vector_index(),
            endpoint_name=self.__config.get_vector_endpoint()
        )
        self.__llm_endpoint = self.__config.get_generator_endpoint()
        self.__llm_api_key = ""  # read from databricks secret
        self.__vector_query_type = self.__config.get_vector_query_type()

    def __get_query_embedding(self, query_text: str) -> List[float]:
        """
        Get the query embedding vector.

        :param query_text: Query text.
        :return: Query embedding vector.
        """
        response = self.__deploy_client.predict(endpoint=self.__config.get_embedding_model(), inputs={"input": query_text})
        return response.data[0]["embedding"]

    def __similarity_search(self, query_text: str, query_filter: dict) -> List[dict]:
        """
        Perform similarity search.

        :param query_text: Query text.
        :param query_filter: Query filter.
        :return: Search results.
        """
        embedding_model = DatabricksEmbeddings(endpoint="databricks-gte-large-en")
        vector_search_as_retriever = DatabricksVectorSearch(
            self.__index,
            text_column="page_content",
            columns=["id", "page_content", "metadata"],
            embedding=embedding_model
        ).as_retriever(search_kwargs={"k": 3})

    def format_context(docs):
        chunk_contents = [f"Passage: {d.page_content}\n" for d in docs]
        return "".join(chunk_contents)

        return vector_search_as_retriever.invoke(query_text)

    def __live_qna(self, query: str, image_collection: List[str]) -> List[str]:
        """
        Perform live QnA on images.

        :param query: Query text.
        :param image_collection: List of image URLs.
        :return: List of image descriptions.
        """
        contexts = []
        for img_url in image_collection:
            if os.path.exists(img_url):
                with open(img_url, "rb") as pil_image:
                    buffer = BytesIO(pil_image.read()).getvalue()
                    img_encoding = base64.b64encode(buffer).decode()
                    gptv4_service = AzureGpt4VService(llm_endpoint=self.__config.get_vision_endpoint(), llm_api_key=self.__llm_api_key)
                    image_descriptions = gptv4_service.image_description(
                        images=[img_url],
                        prompt=f"explain about the image and get information relative to the query: {query}",
                        detail_mode="auto",
                        image_urls=[img_encoding]
                    )
                    contexts.extend(image_descriptions)
        return contexts

    def __get_multi_turn_question(self, query: str, chat_history: List[dict]) -> str:
        """
        Get multi-turn question.

        :param query: Query text.
        :param chat_history: Chat history.
        :return: Multi-turn question.
        """
        conversation_string = """
        history-Q: {question}
        history-A: {answer}
        """
        conversation = [
            conversation_string.format(question=chat.get("query"), answer=chat.get("answer")) for chat in chat_history
        ]
        conversation = "\n".join(conversation)
        prompt = self.__config.get_muliturn_prompt().format(
            conversation=conversation, question=query)
        response = self.__get_openai_response(prompt)
        return response.strip()

    def get_context(self, query: str, query_filter: dict) -> List[dict]:
        """
        Get context for the query.

        :param query: Query text.
        :param query_filter: Query filter.
        :return: Search results.
        """
        return self.__similarity_search(query, query_filter)

    def __get_openai_response(self, user_prompt: str) -> str:
        """
        Get response from OpenAI.

        :param user_prompt: User prompt.
        :return: Response text.
        """
        payload = json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "temperature": 0.2,
            "top_p": 0.95,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "max_tokens": 800
        })
        headers = {
            'api-key': self.__llm_api_key,
            'Content-Type': 'application/json'
        }
        response = requests.post(self.__llm_endpoint, headers=headers, data=payload)
        return response.json()["choices"][0]["message"]["content"]

    def __get_db_instruct_response(self, user_prompt: str) -> str:
        """
        Get response from Databricks instruction model.

        :param user_prompt: User prompt.
        :return: Response text.
        """
        model = ChatDatabricks(
            endpoint="databricks-meta-llama-3-1-405b-instruct",
            extra_params={"temperature": 0.01},
        )
        return model.invoke(user_prompt).content

    def __format_chat_history_for_prompt(self, history: List[dict]) -> List[dict]:
        """
        Format chat history for prompt.

        :param history: Chat history.
        :return: Formatted chat history.
        """
        formatted_chat_history = []
        if history:
            for chat_message in history:
                if chat_message["role"] == "user":
                    formatted_chat_history.append(HumanMessage(content=chat_message["content"]))
                elif chat_message["role"] == "assistant":
                    formatted_chat_history.append(AIMessage(content=chat_message["content"]))
        return formatted_chat_history

    def __get_llm_response(self, prompt, context: str, query: str, chat_history: List[dict]) -> str:
        """
        Get response from LLM.

        :param prompt: Prompt template.
        :param context: Context text.
        :param query: Query text.
        :param chat_history: Chat history.
        :return: Response text.
        """
        input_data_messages = []
        if chat_history:
            for history in chat_history:
                chat_query, chat_answer = history.get("query"), history.get("answer")
                input_data_messages.append({'content': chat_query, 'role': 'user'})
                input_data_messages.append({'content': chat_answer, 'role': 'assistant'})
        chat_history = self.__format_chat_history_for_prompt(input_data_messages)
        updated_prompt = prompt.invoke(
            {
                "chat_history": chat_history,
                "question": query,
                "context": context
            }
        )

        openai = True  # Openai Model
        if openai:
            user_prompt = prompt.format(context=context, question=query, chat_history=chat_history)
            return self.__get_openai_response(user_prompt)
        else:
            return self.__get_db_instruct_response(updated_prompt)

    def predict(self, query: str, chat_history: List[dict] = [], query_filter: dict = {}) -> Dict[str, str]:
        """
        Predict the response for a query.

        :param query: Query text.
        :param chat_history: Chat history.
        :param query_filter: Query filter.
        :return: Response and citations.
        """
        if chat_history:
            query = self.__get_multi_turn_question(query, chat_history)
        results = self.get_context(query, query_filter)
        image_list, contexts, citations = [], [], []
        for context in results:
            contexts.append(context.page_content)
            file_path = context.metadata.get("source")
            citations.append({"content": context.page_content, "source": file_path})
            if file_path and file_path.lower().endswith(".jpeg"):
                image_list.append(file_path)
        if image_list:
            image_context = self.__live_qna(query, image_list)
            contexts += image_context
        final_context = "\n".join(contexts)
        prompt = PromptTemplate(
            template=self.__config.get_generator_prompt(),
            input_variables=["chat_history", "question", "context"],
        )
        result = self.__get_llm_response(prompt, final_context, query, chat_history)
        return {"response": result, "citations": citations, "query": query}


class RagChat(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.chat = Chat()

    def predict(self, context, input_data: pd.DataFrame) -> Dict[str, str]:
        """
        Predict the response for a query using RagChat.

        :param context: Context (not used).
        :param input_data: Input data containing query and chat history.
        :return: Response and citations.
        """
        query = input_data["query"].values[0]
        chat_history = input_data["chat_history"].values[0]
        input_data_messages = []
        if chat_history:
            chat_history = json.loads(chat_history)
            chat_history = chat_history if len(chat_history) <= 2 else chat_history[CHAT_HISTORY_COUNT:]
            for history in chat_history:
                chat_query, chat_answer = history['query'], history['answer']
                input_data_messages.append({'content': chat_query, 'role': 'user'})
                input_data_messages.append({'content': chat_answer, 'role': 'assistant'})
        input_data_messages.append({'content': query, 'role': 'user'})
        input_data = {'messages': input_data_messages}

        return self.chat.predict(query, chat_history, query_filter={})


# COMMAND ----------

# MAGIC %md
# MAGIC #### TEST LOCAL MODEL

# COMMAND ----------

ragChatWrapper = RagChat()
mlflow.models.set_model(model=ragChatWrapper)

chat_history = []
print(dt.now())
query = "Who is the CEO of ATT?"
query = "who is major stock holder of ATT?"
input_df = pd.DataFrame({
    "query": [query],
    "chat_history": [json.dumps(chat_history)]
})
predictions = ragChatWrapper.predict(None, input_df)
print(query)
print(predictions.get("response"))

# COMMAND ----------

# MAGIC %md
# MAGIC #### REGISTER MLFLOW MODEL

# COMMAND ----------


mlflow.set_tracking_uri("databricks")

input_schema = Schema(
    [
        ColSpec(DataType.string, "query"),
        ColSpec(DataType.string, "chat_history")

    ]
)
output_schema = Schema([ColSpec(DataType.string, "response"),ColSpec(DataType.string, "citations"),ColSpec(DataType.string, "query")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

conda_env = {
    "channels": ["defaults"],
    "dependencies": [
        f"python=3.11",
        "pip",
        {
            "pip": [
                f"databricks-sdk==0.33.0",
                f"databricks-vectorsearch==0.41",
                f"langchain_databricks==0.1.0",
            ],
        },
    ],
    "name": "wallstreet_env",
}

run_id = None
mlflow.set_experiment("wallstreet_model")
with mlflow.start_run() as run:
  run_id= run.info.run_id
  mlflow.pyfunc.log_model(artifact_path="wallstreet_model", python_model=ragChatWrapper,#)#,
                          signature=signature)

# COMMAND ----------

# MAGIC %md
# MAGIC #### TEST MLFLOW MODEL

# COMMAND ----------

import mlflow
import pandas as pd
import json
model_uri = f'runs:/{run_id}/wallstreet_model'
print(model_uri)
chat_history = []
loaded_model = mlflow.pyfunc.load_model(model_uri)
stock_chat_history = [{"question":"test answer","answer":"test Answer"}]

stock_chat_history = [{"query":"highest revenue of AT&T","answer":"The highest revenue estimate among these is $130,110,000,000 (130.110 billion USD)."}]
query = "what is the lowest revenue?"

query_filter = {}
input_df = pd.DataFrame({
    "query": [query],
    "chat_history": [json.dumps(stock_chat_history)] 
})
predictions = loaded_model.predict(input_df)
print(predictions.get("response"))
