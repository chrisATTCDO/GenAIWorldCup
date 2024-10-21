# Databricks notebook source
# MAGIC %md
# MAGIC # Validate data for fine-tuning runs
# MAGIC
# MAGIC This notebook shows how to validate your data and ensure data integrity is upheld while using Mosaic AI Model Training. It also provides guidance on how to estimate costs based on token usage during fine-tuning runs.
# MAGIC
# MAGIC This script serves as an ad-hoc utility for you to run independently prior to starting fine-tuning. Its primary function is to validate your data before you invoke the Finetuning API. This script is not meant for use during the training process. 
# MAGIC
# MAGIC The inputs to this validation script are assumed to be the same or a subset of the Mosaic AI Model Training API accepted inputs like the following:

# COMMAND ----------

cfg = {
    model: str,
    train_data_path: str,
    save_folder: str,
    *,
    task_type: Optional[str] = "INSTRUCTION_FINETUNE",
    eval_data_path: Optional[str] = None,
    eval_prompts: Optional[List[str]] = None,
    custom_weights_path: Optional[str] = None,
    training_duration: Optional[str] = None,
    learning_rate: Optional[float] = None,
    context_length: Optional[int] = None,
    experiment_trackers: Optional[List[Dict]] = None,
    disable_credentials_check: Optional[bool] = None,
    timeout: Optional[float] = 10,
    future: Literal[False] = False,
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install libraries

# COMMAND ----------

# MAGIC %pip uninstall -y llm-foundry

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install --upgrade --no-deps git+https://github.com/mosaicml/llm-foundry.git@byod/data_validation
# MAGIC %pip install 'mosaicml[libcloud,wandb,oci,gcs]>=0.23.4,<0.24'
# MAGIC %pip install 'mlflow>=2.14.1,<2.16'
# MAGIC %pip install 'transformers>=4.43.2,<4.44'
# MAGIC %pip install "mosaicml-streaming>=0.8.0,<0.9"
# MAGIC %pip install 'catalogue>=2,<3'
# MAGIC %pip install 'beautifulsoup4>=4.12.2,<5'
# MAGIC %pip install -U datasets
# MAGIC %pip install omegaconf
# MAGIC %pip install einops
# MAGIC %pip install sentencepiece

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import re
import json
import tempfile
import random
import numpy as np
import pandas as pd 
from collections import defaultdict
from argparse import ArgumentParser, Namespace
from typing import cast 

import datasets 

from llmfoundry.utils import (create_om_cfg, token_counts_with_collate, 
        check_HF_datasets, is_hf_dataset_path, is_uc_delta_table,
        integrity_check, convert_text_to_mds, parse_args, plot_hist,
)

from llmfoundry.data.finetuning.tasks import (_validate_chat_formatted_example,
                                              _tokenize_prompt_response_formatted_example,
                                              _get_example_type, ChatFormattedDict, PromptResponseDict )
import transformers
transformers.logging.set_verbosity_error()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Instruction fine-tuning
# MAGIC
# MAGIC In this section, you set up the parameters for the validation notebook. 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The following are the API arguments for fine-tuning tasks:
# MAGIC
# MAGIC | Argument | Description|
# MAGIC |-------|-----|
# MAGIC |`model` | Specifies the model to be used for fine-tuning. For example, `EleutherAI/gpt-neox-20b`|
# MAGIC | `train_data_path`| The path to the training data. It can be either a Hugging Face dataset, a path to a JSON Lines (`.jsonl`) file or a delta table.|
# MAGIC |`task_type`| Defines the type of task for which the training strategy will be applied. It is either `INSTRUCTION_FINETUNE` or `CONTINUED_PRETRAIN`.|
# MAGIC |`training_duration`| The duration of the training process, expressed in numerical terms with units of training epochs.|
# MAGIC |`context_length`| Specifies the context length of the model, set to 2048. This determines how many tokens the model considers for each training example.|
# MAGIC
# MAGIC The following are temporary data path configuration arguments:
# MAGIC
# MAGIC - `temporary_jsonl_data_path`: Defines a file system path where temporary data related to the training process will be stored. You need to make sure the path is not shared by other users on the cluster because sharing could cause problems.
# MAGIC - Environment variables for Hugging Face caches (`HF_DATASETS_CACHE`) are set to `'/tmp/'`, directing dataset caching to a temporary directory.
# MAGIC
# MAGIC
# MAGIC You need to specify context length based on the model. For the latest supported models and their associated context lengths, use the `get_models()` function. 
# MAGIC See supported models ([AWS](https://docs.databricks.com/large-language-models/foundation-model-training/index.html#supported-models)| [Azure](https://learn.microsoft.com/azure/databricks/large-language-models/foundation-model-training/#--supported-models))
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC The following sets up your `home` directory:

# COMMAND ----------

# Make sure you have write access to the ``home`` directory
home = os.path.join('/tmp', 'ift')
os.makedirs(home, exist_ok=True)
os.chdir(home)

# COMMAND ----------

# MAGIC %md
# MAGIC The following defines the fine-tuning API arguments and uses the `temporary_jsonl_data_path` to define the file system path where you store temporary data related to the training process. Environment variables for Hugging Face caches (`HF_DATASETS_CACHE`)  are set to `/tmp/`, which directs dataset caching to a temporary directory.

# COMMAND ----------

FT_API_args = Namespace(
    model= 'mosaicml/mpt-7b', # Other examples: 'EleutherAI/gpt-neox-20b',
    train_data_path= 'mosaicml/dolly_hhrlhf/train',  # Other examples: '/path/to/train.jsonl', 'catalog.schema.table', 'iamroot/chat_formatted_examples/train', 
    task_type='INSTRUCTION_FINETUNE', # 'CHAT_COMPLETION'
    training_duration=3,
    context_length=2048,
)

temporary_jsonl_data_path = os.path.join(home, 'ft_data_11Jan24_3/train')
os.environ['HF_DATASETS_CACHE'] = os.path.join(home, 'hf_cache')
os.makedirs(temporary_jsonl_data_path, exist_ok=True)
os.makedirs(os.environ['HF_DATASETS_CACHE'], exist_ok=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data loading
# MAGIC
# MAGIC The instruction fine-tuning data needs to have the dictionary format below:
# MAGIC
# MAGIC ```
# MAGIC prompt: xxx
# MAGIC response or completion: yyy
# MAGIC ```
# MAGIC
# MAGIC Based on `FT_API_args.train_data_path`, select an ingestion method from one of the following options:
# MAGIC
# MAGIC - A JSONL file which is stored in an object store supported by Composer.
# MAGIC - A Hugging Face dataset ID. For this option, you need to also provide a split. 
# MAGIC - A Delta table. 

# COMMAND ----------

raw_dataset = None

if is_hf_dataset_path(FT_API_args.train_data_path):
    check_HF_datasets(FT_API_args.train_data_path)
    dataset_id, split = '/'.join(FT_API_args.train_data_path.split('/')[:2]), FT_API_args.train_data_path.split('/')[-1]    
    raw_dataset = datasets.load_dataset(dataset_id, split=split)       
else:
    if is_uc_delta_table(FT_API_args.train_data_path):    
        df = spark.read.table(FT_API_args.train_data_path).toPandas()
        df.to_json(os.path.join(temporary_jsonl_data_path, 'data.jsonl'), orient='records', lines=True)
        raw_dataset = datasets.Dataset.from_pandas(df) 
        FT_API_args.train_data_path = temporary_jsonl_data_path
    else: 
        # train_data_path is a jonsl file (local/remote)
        from composer.utils import dist, get_file, parse_uri 
        data_path = FT_API_args.train_data_path 
        backend, _, _ = parse_uri(data_path)
        if backend not in ['', None]: # It's a remote path, download before loading it
            with tempfile.TemporaryDirectory() as tmp_dir:
                destination = os.path.join(tmp_dir, 'data.jsonl')
                get_file(data_path, destination)
                df = pd.read_json(destination, orient='records', lines=True)    
        else: 
            df = pd.read_json(data_path, orient='records', lines=True)    

        raw_dataset = datasets.Dataset.from_pandas(df)
        FT_API_args.train_data_path = os.path.dirname(data_path)

if raw_dataset is None: 
    raise RuntimeError("Can't find a proper ingestion method")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data quality checks on the dataset

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC This section of the notebook performs a series of checks on the initial dataset to ensure its quality and expected format. This process ensures that the dataset adheres to the expected structure and contains the necessary keys for further processing. The checks are outlined below.
# MAGIC
# MAGIC 1. The total number of examples in the dataset is printed.
# MAGIC 2. The first example from the dataset is displayed. This provides a quick glimpse into the data structure and format.
# MAGIC 3. Data format validation:
# MAGIC - The dataset is expected to consist of dictionary-like objects (for example, key-value pairs). 
# MAGIC - A check is performed to validate this structure.
# MAGIC 4. Key presence validation:
# MAGIC - Allowed prompt and response keys, chat roles are defined in [llmfoundry](https://github.com/mosaicml/llm-foundry/blob/main/llmfoundry/data/finetuning/tasks.py): _ALLOWED_RESPONSE_KEYS and _ALLOWED_PROMPT_KEYS and _ALLOWED_ROLES.
# MAGIC - For prompt response dataset, the script checks for the presence of at least one prompt key and one response key in each example.
# MAGIC   - Prompt validation: Each example is checked for the presence of keys defined in _ALLOWED_PROMPT_KEYS. If no valid prompt key is found, it is counted as a format error. 
# MAGIC   - Response validation: Similarly, each example is checked for the presence of keys defined in _ALLOWED_RESPONSE_KEYS. An absence of a valid response key is also counted as a format error.
# MAGIC - For chat formatted dataset, the script checks if the message content is formatted valid by calling [_validate_chat_formatted_example](https://github.com/mosaicml/llm-foundry/blob/cffd75e94e5c53b1b14c67cd17e0916fecfd0e16/llmfoundry/data/finetuning/tasks.py#L130) helper function.
# MAGIC
# MAGIC If any format errors are found during the checks, they are reported. A summary of errors is printed, categorizing them into types like `data_type` (non-dictionary data), `missing_prompt`, and `missing_response`.
# MAGIC
# MAGIC If no errors are found, a congratulatory message is displayed, indicating that all checks have passed successfully.

# COMMAND ----------

# Initial dataset stats
print("Num examples:", len(raw_dataset))
print("First example:")
for ex in raw_dataset: 
    print(ex)
    print() 
    break 

format_errors = defaultdict(int)

for example in raw_dataset:
    try: 
        example_format = _get_example_type(ex)
    except ValueError:
        format_errors["unknown example type"] += 1 
        continue 

    if example_format == 'chat':
        try: 
            chat_example = cast(ChatFormattedDict, example)
            _validate_chat_formatted_example(chat_example)
        except Exception as e:             
            format_errors['chat_format_error'] += 1  
            print(e)
            break 

    elif example_format == 'prompt_response':
        try:
            prompt_response_example: PromptResponseDict = cast(
                PromptResponseDict, example)
        except Exception as e: 
            format_errors['prompt_response_format_error'] += 1  
            print(e)
            break 
        
if format_errors:
    print("Oops! Found errors:")
    for k, v in format_errors.items():
        print(f"{k}: {v}")
else:
    print("Congratulations! No errors found")     

# COMMAND ----------

# MAGIC %md
# MAGIC ### Token estimation
# MAGIC
# MAGIC Tokenize the raw dataset and get some statistics of the tokens. By doing this, you can estimate the overall cost based on a default trainining run duration. You iterate over the dataloader and sum the number of tokens from each batch.

# COMMAND ----------

n_epochs = FT_API_args.training_duration if FT_API_args.training_duration is not None else 1 
batch_tokens = token_counts_with_collate(FT_API_args)
n_billing_tokens_in_dataset = sum(batch_tokens['ntokens'])

# COMMAND ----------

# MAGIC %md
# MAGIC The fine-tuning API internally ingests the dataset and runs tokenization with the selected tokenizer. 
# MAGIC The output dataset is a collection of samples and each sample is a collection of token IDs represented as integers.  
# MAGIC
# MAGIC A histogram is generated so you can visualize the distribution of the frequency of token counts in samples in the dataset. 
# MAGIC The visualization aids in identifying patterns, outliers, and central tendencies in the token distribution.

# COMMAND ----------

print(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be used by the model during training")
print(f"Assume you'll train for {n_epochs} epochs on this dataset")
print(f"Then ~{n_epochs * n_billing_tokens_in_dataset} tokens will be running through the model during training")
plot_hist(pd.Series(batch_tokens['ntokens']))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Continued pretraining

# COMMAND ----------

# MAGIC %md
# MAGIC Similar to instruction fine-tuning, you need to specify the following arguments: 
# MAGIC   
# MAGIC | Argument | Description|
# MAGIC |-------|-----|
# MAGIC |`model` | Specifies the model to be used for fine-tuning. For example, `EleutherAI/gpt-neox-20b`|
# MAGIC | `train_data_path`| The path to the training data. Currently, only a remote or local path to a collection of .txt files is supported|
# MAGIC |`task_type`| Defines the type of task for which the training strategy is applied. It is either `INSTRUCTION_FINETUNE` or `CONTINUED_PRETRAIN`.|
# MAGIC |`training_duration`| The duration of the training process, expressed in numerical terms with units of training epochs.|
# MAGIC |`context_length`| Specifies the context length of the model, set to 2048. This determines how many tokens the model considers for each training example. For continued pretraining, tokens are concatenated to form samples of length equal to `context_length`|
# MAGIC
# MAGIC The following are temporary data path configuration arguments:
# MAGIC
# MAGIC - temporary_mds_output_path: Defines a filesystem path where a notebook that's running data can be stored. You need to make sure the path isn't shared by other users on the cluster because sharing could cause problems. For example, you can make it distinguishable by adding your username to the path.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The following defines the fine-tuning API arguments and uses the `temporary_mds_output_path` to define the file system path where temporary data related to the training process is stored. 

# COMMAND ----------

FT_API_args = Namespace(
    model= 'mosaicml/mpt-7b',
    train_data_path= '/Volumes/main/mosaic_hackathon/managed-volume/ABT',
    task_type='CONTINUED_PRETRAIN',
    training_duration=3,
    context_length=2048,
)
temporary_mds_output_path = '/Volumes/main/mosaic_hackathon/managed-volume/mds_data_11Jan24_5'

# COMMAND ----------

# MAGIC %md
# MAGIC Generate a synthetic dataset. Replace train_data_path with your raw data path in practice.

# COMMAND ----------

def generate_synthetic_dataset(folder_path, num_files=128):
    """Generate a synthetic dataset of text files with random words."""
    def generate_random_words(num_words=50):
        words = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew", "kiwi", "lemon", "mango", "nectarine", "orange", "papaya", "quince", "raspberry", "strawberry", "tangerine", "ugli", "vanilla", "watermelon", "xigua", "yam", "zucchini"]
        return ' '.join(random.choice(words) for _ in range(num_words))

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    for i in range(num_files):
        file_path = os.path.join(folder_path, f"file_{i}.txt")
        with open(file_path, 'w') as file:
            file.write(generate_random_words())

    print(f"Generated {num_files} files in '{folder_path}'.")

generate_synthetic_dataset(FT_API_args.train_data_path)

# COMMAND ----------

## Run the following to remove files from this temporary path
!rm -rf {temporary_mds_output_path}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ingestion, tokenization and materialization
# MAGIC
# MAGIC Continued pre-training accepts a folder of `.txt` files as input data. It tokenizes the text fields and materializes them as a streaming dataset of MDS format. 
# MAGIC
# MAGIC Mosaic AI Model Training uses [llmfoundry/scripts/data_prep/convert_text_to_mds.py](https://github.com/mosaicml/llm-foundry/blob/main/scripts/data_prep/convert_text_to_mds.py) to download all the `.txt` files and convert them to MDS. 
# MAGIC
# MAGIC This notebook provides two additional approaches using Spark and Dask. 
# MAGIC
# MAGIC Continued pre-training datasets are normally much larger than instruction fine-tuning, so the tokenization and materialization can be very time consuming. 

# COMMAND ----------

import os
os.makedirs(temporary_mds_output_path, exist_ok=True)

# COMMAND ----------

cfg, tokenizer = create_om_cfg(FT_API_args)

input_folder = FT_API_args.train_data_path
output_folder = temporary_mds_output_path
concat_tokens = FT_API_args.context_length
tokenizer_name = FT_API_args.model

# Run convert_text_to_mds.py and dump MDS dataset to "save_folder"
args = parse_args(tokenizer_name, concat_tokens, output_folder, input_folder)

n_samples = convert_text_to_mds(
    tokenizer_name=args.tokenizer,
    output_folder=args.output_folder,
    input_folder=args.input_folder,
    concat_tokens=args.concat_tokens,
    eos_text=args.eos_text,
    bos_text=args.bos_text,
    no_wrap=args.no_wrap,
    compression=args.compression,
    processes=1,
    reprocess=True,
    args_str=str(args), 
    trust_remote_code=False)

n_billing_tokens_in_dataset = n_samples * concat_tokens

# COMMAND ----------

# MAGIC %md
# MAGIC ### Token estimation

# COMMAND ----------

MAX_TOKENS_PER_EXAMPLE = FT_API_args.context_length if FT_API_args.context_length is not None else 4096
TARGET_EPOCHS = FT_API_args.training_duration if FT_API_args.training_duration is not None else 1 
n_epochs = TARGET_EPOCHS

print(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training")
print(f"By default, you'll train for {n_epochs} epochs on this dataset")
print(f"By default, ~{n_epochs * n_billing_tokens_in_dataset} tokens will be used in training")