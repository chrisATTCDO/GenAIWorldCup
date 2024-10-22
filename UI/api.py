import requests  
import pandas as pd  
import json  
from config import PAT, URL  # Import PAT and URL from config.py  
  
# Function to create TensorFlow Serving JSON input  
def create_tf_serving_json(data):  
    # Check if data is a dictionary and convert it to JSON format  
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}  
  
# Function to score the model using the given dataset  
def score_model(dataset):  
    # Set up the headers for the HTTP request, including the authorization token  
    headers = {'Authorization': f'Bearer {PAT}', 'Content-Type': 'application/json'}  
      
    # Prepare the data for JSON serialization  
    ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)  
    data_json = json.dumps(ds_dict, allow_nan=True)  
      
    # Send a POST request to the model serving endpoint  
    response = requests.post(headers=headers, url=URL, data=data_json)  
      
    # Check if the request was successful  
    if response.status_code != 200:  
        # Raise an exception if the request failed  
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')  
      
    # Return the JSON response from the server  
    return response.json()  
