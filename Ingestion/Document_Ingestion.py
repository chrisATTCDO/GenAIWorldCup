# Databricks notebook source
# %pip install langchain_community pymupdf openai databricks-vectorsearch jq unstructured
# dbutils.library.restartPython()

# COMMAND ----------

import os
import json
import uuid
import time
import logging as logger
import io
import base64
import urllib.parse
import platform
import unicodedata
import re
from io import BytesIO
from typing import List, Any, Optional

import fitz
import pandas as pd
import requests
from PIL import Image
import openai
import mlflow
import mlflow.deployments
from databricks.vector_search.client import VectorSearchClient
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    CSVLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import JSONLoader
from langchain_core.documents import Document

# COMMAND ----------

class ExtensionLoader:
    """
    A class to load extension configurations.

    Attributes:
    ----------
    loader_class : str
        The class name of the file loader.
    loader_kwargs : dict
        The keyword arguments for the loader.
    splitter_class : str
        The class name of the splitter.
    splitter_kwargs : dict
        The keyword arguments for the splitter.

    Methods:
    -------
    __init__(ingestion_data: dict):
        Initializes the ExtensionLoader with ingestion data.
    """
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
    """
    A class to handle configuration settings.

    Attributes:
    ----------
    __domain_data : dict
        The domain data loaded from the configuration file.
    __vector_store : dict
        The vector store configuration.
    __generator : dict
        The generator configuration.
    __vision : dict
        The vision configuration.
    __prompt : dict
        The prompt configuration.
    __ingestion_configuration : dict
        The ingestion configuration.
    __extension_configs : dict
        The extension configurations.

    Methods:
    -------
    __init__(file_path: str):
        Initializes the Config with a file path.
    __load_extension_list() -> dict:
        Loads the extension list from the ingestion configuration.
    get_loader_for_extension(extension: str) -> Optional[ExtensionLoader]:
        Gets the loader for a specific extension.
    get_embedding_model() -> str:
        Gets the embedding model name.
    get_embedding_model_dimension() -> int:
        Gets the embedding model dimension.
    get_vector_index_schema() -> dict:
        Gets the vector index schema.
    get_vector_index_primary_key() -> str:
        Gets the vector index primary key.
    get_vector_index_vector_column() -> str:
        Gets the vector index embedding vector column.
    get_vector_endpoint() -> str:
        Gets the vector endpoint name.
    get_vector_index() -> str:
        Gets the vector index name.
    get_generator_endpoint() -> str:
        Gets the generator endpoint name.
    get_vision_endpoint() -> str:
        Gets the vision endpoint name.
    get_generator_model() -> str:
        Gets the generator model name.
    get_vector_query_type() -> str:
        Gets the vector query type.
    get_generator_prompt() -> str:
        Gets the generator prompt.
    get_symbol_identifier_prompt() -> str:
        Gets the symbol identifier prompt.
    get_symbol_conversation_prompt() -> str:
        Gets the symbol conversation prompt.
    get_multiturn_prompt() -> str:
        Gets the multi-turn prompt.
    """
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
        self.__vision = self.__domain_data.get("vision", None)
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

    def get_loader_for_extension(self, extension: str) -> Optional[ExtensionLoader]:
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

    def get_vision_endpoint(self) -> str:
        """
        Get the vision endpoint name.

        :return: Vision endpoint name.
        :raises Exception: If vision endpoint is not present.
        """
        if self.__vision:
            endpoint_name = self.__vision.get("openai_endpoint")
            if endpoint_name:
                return endpoint_name
        raise Exception("Vision endpoint not present")

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
            multiturn_prompt = self.__prompt.get("multiturn_prompt")
            if multiturn_prompt:
                return multiturn_prompt
        raise Exception("Multi-turn prompt not present")

# COMMAND ----------

class EmbeddingGenerator:
    """
    A class to generate embeddings from a model deployed on Azure Databricks.

    Attributes:
    ----------
    deploy_client : mlflow.deployments.DeployClient
        The deployment client for interacting with the model.
    endpoint : str
        The endpoint URL of the deployed model.

    Methods:
    -------
    __init__(endpoint: str):
        Initializes the EmbeddingGenerator with the given endpoint.
    generate_embeddings(text: str) -> List[float]:
        Generates embeddings for the given text.
    """
    def __init__(self, endpoint: str):
        """
        Initialize the EmbeddingGenerator with the given endpoint.

        :param endpoint: The endpoint URL of the deployed model.
        """
        self.deploy_client = mlflow.deployments.get_deploy_client("databricks")
        self.endpoint = endpoint

    def generate_embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings for the given text.

        :param text: The input text to generate embeddings for.
        :return: A list of floats representing the embeddings.
        :raises Exception: If there is an error generating embeddings.
        """
        try:
            response = self.deploy_client.predict(endpoint=self.endpoint, inputs={"input": text})
            embeddings = response.data[0]["embedding"]
            return embeddings
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")


# COMMAND ----------

class VectorSearchManager:
    """
    A class to manage vector search operations.

    Attributes:
    ----------
    __config : Config
        The configuration object.
    client : VectorSearchClient
        The client for interacting with the vector search service.
    index_name : str
        The name of the vector index.
    endpoint_name : str
        The endpoint name of the vector search service.
    embedding_dimension : int
        The dimension of the embedding vectors.
    primary_key : str
        The primary key for the vector index.
    embedding_vector_column : str
        The column name for the embedding vectors.
    index : Optional[dict]
        The vector index object.

    Methods:
    -------
    __init__(config: Config):
        Initializes the VectorSearchManager with the given configuration.
    index_exists() -> bool:
        Checks if the index exists.
    get_index() -> Optional[dict]:
        Gets the vector index object.
    create_index():
        Creates the vector database index.
    add_documents(documents: List[dict]) -> bool:
        Adds the documents to the vector database.
    """
    def __init__(self, config: Config):
        """
        Initialize the VectorSearchManager with the given configuration.

        :param config: The configuration object.
        """
        self.__config = config
        self.client = VectorSearchClient(disable_notice=True)
        self.index_name = self.__config.get_vector_index()
        self.endpoint_name = self.__config.get_vector_endpoint()
        self.embedding_dimension = self.__config.get_embedding_model_dimension()
        self.primary_key = self.__config.get_vector_index_primary_key()
        self.embedding_vector_column = self.__config.get_vector_index_vector_column()
        self.index = None
        if not self.index_exists():
            self.create_index()
        else:
            self.index = self.client.get_index(
                index_name=self.index_name,
                endpoint_name=self.endpoint_name
            )

    def index_exists(self) -> bool:
        """
        Checks if the index exists.

        :return: True if the index exists, False otherwise.
        """
        try:
            existing_index = self.client.list_indexes(self.endpoint_name)
            return self.index_name in [index["name"] for index in existing_index["vector_indexes"]]
        except Exception as e:
            return False

    def get_index(self):
        """
        Get the vector index object.

        :return: The vector index object.
        """
        return self.index

    def create_index(self):
        """
        Creates the vector database index.

        :raises Exception: If there is an error creating the index.
        """
        schema = self.__config.get_vector_index_schema()
        filter_columns = []  # Define filter_columns if needed
        for col in filter_columns:
            schema[col] = "string"
        if self.index_exists():
            print(f"Index {self.index_name} already exists")
            return schema
        try:
            self.index = self.client.create_direct_access_index(
                endpoint_name=self.endpoint_name,
                primary_key=self.primary_key,
                index_name=self.index_name,
                embedding_dimension=self.embedding_dimension,
                embedding_vector_column=self.embedding_vector_column,
                schema=schema
            )
            print(f"Created index {self.index_name}")
        except Exception as e:
            if "Vector index" in str(e) and "is not ready" in str(e):
                print(f"Index {self.index_name} is not ready. Retrying...")
                time.sleep(10)  # Wait for 10 seconds before retrying
                self.create_index()
            else:
                raise Exception(f"Error creating index {self.index_name}: {str(e)}")

    def add_documents(self, documents: List[dict]) -> bool:
        """
        Adds the documents to the vector database.

        :param documents: List of documents to be added.
        :return: True if documents are added successfully, False otherwise.
        :raises Exception: If there is an error adding documents.
        """
        try:
            index = self.client.get_index(self.endpoint_name, self.index_name)
            upload_doc = index.upsert(documents)
            if upload_doc["status"] == "SUCCESS":
                print(f"{len(documents)} documents added to index {self.index_name}")
                return True
            else:
                raise Exception(f"Error adding documents to index {self.index_name}: {str(upload_doc)}")
        except Exception as e:
            raise Exception(f"Error adding documents to index {self.index_name}: {str(e)}")

# COMMAND ----------

class ImageExtract:
  """
  A class to extract images from a PDF file, save them to a specified directory, and return metadata about the images.

  Attributes:
  ----------
  file_path : str
      The path to the PDF file from which images will be extracted.
  image_path_prefix : str
      A prefix for naming the extracted image files based on the PDF file name.

  Methods:
  -------
  _sanitize_filename(filename: str) -> str:
      Sanitizes a filename to be OS-compatible and safe for saving.
  _convert_url_to_filename(image_url) -> str:
      Converts a URL to a sanitized filename.
  get_image_path(image_url, dir_destination_file) -> str:
      Constructs the full path for saving an image.
  save_image(url: str, image_base64: str, dir_destination_file: str) -> str:
      Saves a base64-encoded image to the specified directory.
  load_file() -> List[Document]:
      Extracts images from the PDF, saves them, and returns metadata about the images.
  """
  def __init__(self, file_path: str):
    self.file_path = file_path
    self.image_path_prefix =  os.path.basename(file_path).replace(".","-").lower()
    
  def _sanitize_filename(self, filename: str) -> str:
        os_name = platform.system()
        filename = urllib.parse.unquote(filename)
        if os_name != "Windows":
            filename = unicodedata.normalize('NFKC', filename)
        else:
            # This is mainly for window machines used by developers
            filename = unicodedata.normalize('NFKD', filename).encode('ascii', 'ignore').decode('ascii')
        filename = re.sub(r'[^\w\s.-]', '', filename.lower())
        return filename 

  def _convert_url_to_filename(self, image_url) -> str:
        image_url = image_url.replace("#unknown-", "/")
        path = urllib.parse.urlparse(image_url).path
        filename = os.path.basename(path)
        sanitized_filename = self._sanitize_filename(filename)
        return sanitized_filename 

  def get_image_path(self, image_url, dir_destination_file) -> str:
          image_name = self._convert_url_to_filename(image_url)

          image_file_path = os.path.join(
              dir_destination_file, "image", f"{image_name}")
          
          return image_file_path    

  def save_image(self, url: str, image_base64: str, dir_destination_file: str) -> str:
        image_dir = os.path.join(dir_destination_file, "image")
        os.makedirs(image_dir, exist_ok=True)
        image_path = self.get_image_path(url, dir_destination_file)
        image_bytes = base64.b64decode(image_base64)

        with open(image_path, 'wb') as file:
            file.write(image_bytes)
    
        return image_path    
      
  def load_file(self):
        doc = fitz.open(self.file_path)
        dir_destination_file = os.path.dirname(
            self.file_path).replace("pending", "images")
        image_collection = {}
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            images = page.get_images(full=True)
            for img_index, img_info in enumerate(images):
                img_index += 1
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                if base_image:
                    image_bytes = base_image["image"]
                    image = Image.open(BytesIO(image_bytes))
                    if image.getbbox() and image.size[0] > 100 and image.size[1] > 100:  # Check if the image is not blank and larger than 50x50
                        buffer = BytesIO(base_image["image"]).getvalue()
                        encoded_image = base64.b64encode(buffer).decode()
                        image_file_path = os.path.join(
                            dir_destination_file, "image", f"{os.path.basename(self.file_path).lower()}_image_page{page_num + 1}_img_nmbr{img_index}.{base_image['ext']}")
                        image_collection[image_file_path] = encoded_image
                        self.save_image(image_file_path, encoded_image, dir_destination_file=dir_destination_file)
        return [Document(
                    page_content="",
                    metadata={
                        "source": self.file_path,
                        "image_collection": image_collection
                    })]

# COMMAND ----------

class TableExtract:
  """
  A class to extract tables from a PDF file and return them as a list of Document objects.

  Attributes:
  ----------
  file_path : str
      The path to the PDF file from which tables will be extracted.

  Methods:
  -------
  load_file() -> List[Document]:
      Extracts tables from the PDF, converts them to CSV format, and returns them as a list of Document objects with metadata.
  """
  def __init__(self, file_path: str):
    self.file_path = file_path

  def load_file(self) -> List[Document]:
          doc = fitz.open(self.file_path)
          current_page_num = None  # For testing
          df_final = {}
          df_list = []
          for page_num in range(doc.page_count):
              page = doc[page_num]
              tables = page.find_tables(
                  horizontal_strategy="text", vertical_strategy="text")
              page_tables_data = []
              metadata = {
                  "source": self.file_path,
                  "file_path": self.file_path,
                  "total_pages": len(doc),
              }
              for table_num, table in enumerate(tables):
                  original_result = table.extract()
                  df = pd.DataFrame(table.extract())
                  df_csv = df.to_csv(sep='|', index=False, header=False)
                  page_tables_data.append(df_csv)
              df_final[f"Page_{page_num + 1}"] = page_tables_data
              df_list.append(Document(page_content=str(
                  page_tables_data), metadata=metadata))
          return df_list

# COMMAND ----------

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
    image_description(images: list, prompt: str, detail_mode: str, image_urls: list = None, deployment_name: Optional[str] = None, llm_endpoint: Optional[str] = None, llm_apAzureGpt4VServicei_version: Optional[str] = None) -> list:
        Generates descriptions for a list of images using the GPT-4V model.
    """
    def __init__(self, llm_endpoint: Optional[str] = None,llm_api_key: Optional[str] = None): 
        self.__llm_endpoint = llm_endpoint
        self.__llm_api_key = "#####"

    def get_image_format(self, base64_source: str):
        image_stream = BytesIO(base64.b64decode(base64_source))
        image = Image.open(image_stream)
        image_format = image.format
        return image_format

    def image_description(
        self,
        images: list,
        prompt: str,
        detail_mode: str,
        image_urls: list = None,
        deployment_name: Optional[str] = None,
        llm_endpoint: Optional[str] = None,
        llm_api_version: Optional[str] = None
    ):
        self.__init__(llm_endpoint, llm_api_version)
        messages = []
        messages.append({"role": "system", "content": prompt})
        documents = []
        content = []
        for i, image in enumerate(image_urls):
            format = self.get_image_format(image).lower()
            if image:
                content.append({"type": "text", "text": f"!()[{image_urls[i][0]}]"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/{format};base64,{image}", "detail": detail_mode}})
            messages.append({"role": "user", "content": content})
            payload = json.dumps({"messages": messages, "enhancements": {"ocr": {"enabled": False}, "grounding": {"enabled": True}}, "temperature": 0.1, "max_tokens": 1000})
            headers = {
                'api-key': self.__llm_api_key,
                'Content-Type': 'application/json'
            }
            response = requests.request("POST", llm_endpoint, headers=headers, data=payload)
            response = json.loads(response.text)
            if response["choices"][0]["message"]["role"] == "assistant":
                documents.append(Document(page_content=response['choices'][0]['message']['content'], metadata={"source": images[0]}))
        return documents

# COMMAND ----------

deploy_client = mlflow.deployments.get_deploy_client("databricks")
class Upload:
  """
  A class to handle the uploading of documents to a vector database.

  Attributes:
  ----------
  file_path : List[str]
      List of file paths to be processed.
  file_name : str
      Name of the file to be processed.
  config_path : str
      Path to the configuration file.
  vector_search_manager : VectorSearchManager
      Instance of the VectorSearchManager to manage vector search operations.

  Methods:
  -------
  document_exists(file_path: str) -> bool:
      Checks if a document already exists in the vector database.
  process_documents() -> list:
      Processes and uploads documents to the vector database.
  """
  def __init__(
        self,
        file_path: str,
        file_name: str,
        config_path: str,
        filter_param: dict = {}                                                                                                
    ):
    self.__config = Config(config_path)
    self.__file_path = file_path
    self.__file_name = file_name   
    self.__embedding_generator = EmbeddingGenerator(endpoint=self.__config.get_embedding_model()) #Instance of the EmbeddingGenerator to generate embeddings for documents.    
    
    self.vector_search_manager = VectorSearchManager(self.__config)
    self.filter_param = filter_param

  def document_exists(self) -> bool:
    try:
        if not self.vector_search_manager.index_exists():
            return False
        else:
            query_text="*"            
            query_vector = self.__embedding_generator.generate_embeddings(query_text)
            index = self.vector_search_manager.get_index()
            res = index.similarity_search(query_vector=query_vector,columns=["page_content","content_vector"],filters={"source": self.__file_path})
            result = res.get("result")
            return True if result and result.get("data_array") else False
    except Exception as e:
        raise Exception(f"Exception: {str(e)}")

  def process_documents(self):
            """
            Uploads the documents to the vector database.
            """
            image_descriptions = []
            file_result = {"file_name":self.__file_name}
            if self.document_exists():
                file_result["status"] = "skipped"
                file_result["error"] = "Document already present"
                return file_result
            _, file_extension = os.path.splitext(self.__file_name)
            file_extension = file_extension[1:]
            loader_config =  self.__config.get_loader_for_extension(file_extension)
        # try:
            if loader_config:
                loader = globals()[loader_config.loader_class](file_path=self.__file_path,**loader_config.loader_kwargs)
                documents = loader.load_and_split()
                splitter = globals()[loader_config.splitter_class](**loader_config.splitter_kwargs)
                documents = splitter.split_documents(documents)
                if file_extension == "pdf":
                    #### Image description from PDF
                    try:
                        image_extract = ImageExtract(self.__file_path)
                        pdf_images = image_extract.load_file()
                        image_collection = pdf_images[0].metadata["image_collection"]
                        img_urls = list(image_collection.keys())
                        img_encodings = list(image_collection.values())
                        for img_url, img_encoding in zip(img_urls, img_encodings):
                            gptv4_service = AzureGpt4VService()
                            image_descriptions = gptv4_service.image_description(
                                images=[img_url],
                                prompt="What is this image about?",
                                detail_mode="auto",
                                image_urls=[img_encoding],
                                llm_endpoint=self.__config.get_vision_endpoint()                    
                            )
                            documents.extend(image_descriptions)
                    except Exception as e:
                        print(f"Error extracting image descriptions from PDF: {e}")                       
                    ### Table Extraction from PDF
                    try:
                        table_extraction = TableExtract(self.__file_path)
                        tables = table_extraction.load_file()
                        documents.extend(tables)
                    except Exception as e:
                        print(f"Error extracting tables from PDF: {e}")

                ## Data to store in vectorDB database
                processed_documents = []
                for page_num, doc in enumerate(documents, 1):                   
                    embeddings = self.__embedding_generator.generate_embeddings(doc.page_content)
                    if not doc.metadata:
                        doc.metadata["page_number"] = str(page_num)
                        doc.metadata["source"] = str(self.__file_path)                    
                    for col in self.filter_param:
                        doc.metadata[col] = doc.metadata.get(col, self.filter_param.get(col, ""))
                    processed_documents.append({"id": f'{uuid.uuid4()}', "page_content": doc.page_content, "source": str(self.__file_path), "metadata": json.dumps(doc.metadata), "content_vector": embeddings, **{col: doc.metadata[col] for col in self.filter_param}})
                print(f"Uploading {len(processed_documents)} documents to vector database")
                self.vector_search_manager.add_documents(processed_documents)
                file_result["status"] = "success"
                file_result["documents"] = processed_documents            
            else:
                file_result["status"] = "failure"
                file_result["error"] = f"Unsupported file format:{self.__file_name}"
        # except Exception as e:
        #     file_result["status"] = "failure"
        #     file_result["error"] = str(e)
            return file_result

# COMMAND ----------

# MAGIC %md
# MAGIC #### DATA INGESTION

# COMMAND ----------

import os
import json
from datetime import datetime as dt
config_path = "/config/wallstreet_config.json"
org_folder_path = '' # Folder Path

total_files = 0
success_files = 0
failure_files = 0
skipped_files = 0
process_start_time = dt.now()    
for files in os.listdir(org_folder_path):
    start_time = dt.now()
    file_path = os.path.join(org_folder_path, files)
    total_files += 1
    print(f"Processing {files}")
    upload_list = Upload(file_path=file_path, file_name=files, config_path=config_path, filter_param={})
    result = upload_list.process_documents()
    if result["status"] == "success":
        success_files += 1
        print(f"Success: {files}")
    elif result["status"] == "skipped":
        skipped_files += 1
        print(f"Skipped: {files}")
    else:
        failure_files += 1
        print(result["error"])
end_time = dt.now()
total_time =  (end_time - start_time).total_seconds()
print(f"COMPANY Time Taken {total_time} seconds")
print(f"Total files: {total_files}, Success files: {success_files}, Skipped files: {skipped_files}, Failure files: {failure_files}")
total_time = (dt.now() - process_start_time).total_seconds()
print(f"TOTAL PROCESSING TIME: {total_time} seconds")
