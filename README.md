# GenAI_Charbot1
Creating a chatbot using llama2,langchain,pinecone vector db

# bash command
``` bash
 python -m venv mchatbot
 ```

``` bash
source ./mchatbot/Scripts/activate
```

```bash
git config  user.email "email.com"
git commit -m "added requirements"
git push -u origin main
```

``` bash
pip install -r requirements.txt
```

## Download the Llama 2 Model:

llama-2-7b-chat.ggmlv3.q4_0.bin


## From the following link:
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main

# Optional
python -m pip install --upgrade pip


# dimension or embedding model in huggingface
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
 with 384 dimension


 # create environment variable
 create .env file and store Pinecone_API_KEY= "*******"
Pinecone_API_env= "******" or "us-east-1"

# running app

``` bash
python store_index.py
```

``` bash
# Finally run the following command
python app.py
```

``` bash 
open up localhost:
```

# tech stack used
Python
LangChain
Flask
Meta Llama2
Pinecone