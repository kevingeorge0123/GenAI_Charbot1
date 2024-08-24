from flask import Flask,render_template,jsonify,request
from dotenv import load_dotenv
from src.helper import download_hugging_face_embeddings
import os
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from src.prompt import prompt_template



app = Flask(__name__)

load_dotenv()

Pinecone_API_KEY_=os.getenv("Pinecone_API_KEY")
Pinecone_API_env_= os.getenv("Pinecone_API_env")

embeddings = download_hugging_face_embeddings()


pc = Pinecone(api_key=Pinecone_API_KEY_)
index_name = "med-chatbot2"
index = pc.Index(index_name)
index.describe_index_stats()


PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}
llm=CTransformers(model="./model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})

vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings,namespace="ns1")
retriever = vectorstore.as_retriever(search_kwargs={'k': 2})

qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever,
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get",methods=["GET","POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])


if __name__ =='__main__':
    app.run(host="0.0.0.0",port=8080, debug =True)




