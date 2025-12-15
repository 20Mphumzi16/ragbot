# Setup

# create vertual environment

python -m venv venv
./venv/Scripts/activate

# install requirements

pip install -r .\requirements.txt

# getting your models

Download Ollama

ollama pull mistral # model
ollama pull mxbai-embed-large # embedding model

# pip install

pip install langchain-community pypdf python-docx
