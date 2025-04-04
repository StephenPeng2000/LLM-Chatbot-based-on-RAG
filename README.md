LLM Chatbot Based on Retrieval-Augmented Generation (RAG)

This project implements a Chatbot leveraging Retrieval-Augmented Generation (RAG) techniques integrated with Large Language Models (LLM). It enhances chatbot interactions by retrieving external knowledge and generating informed, contextually accurate responses.

Project Structure

data/database_dir/Xiaomi_Car/: Contains documents related to Xiaomi's automotive information, serving as the knowledge base.

database.py: Handles knowledge base operations, including loading, storage, and query execution.

emb_model.py: Loads and operates embedding models to convert text into vector representations for similarity retrieval.

knowledge_extract.py: Extracts key information from input texts for subsequent retrieval and generation.

load_documents.py: Responsible for loading and preprocessing documents into the knowledge base.

local_llm_model.py: Interfaces with a local Large Language Model to perform generative dialogue.

web_api.py: Provides a Flask-based Web API for external interaction with the chatbot.

test.py: Contains testing scripts to validate the functionality and performance of each module.

requirement.txt: Lists Python dependencies and libraries required by the project.

Installation and Running

Step 1: Clone the Repository

git clone https://github.com/StephenPeng2000/LLM-Chatbot-based-on-RAG.git

Step 2: Install Dependencies

pip install -r requirement.txt

Step 3: Launch the Web API

python web_api.py

Once launched, you can interact with the chatbot through HTTP requests.

Example Usage

You can interact with the chatbot using HTTP requests. Here's an example using Python's requests library:

import requests

url = 'http://127.0.0.1:5000/chat'
data = {'message': 'What is the latest model of Xiaomi car?'}
response = requests.post(url, json=data)
print(response.json())

This example queries the chatbot for information on Xiaomi's latest car model, leveraging both the embedded knowledge base and generative capabilities.

Contributing

Contributions and suggestions are welcome. Feel free to submit issues or pull requests.

License

This project is licensed under the MIT License. See the LICENSE file for details.
