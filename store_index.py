import os
from dotenv import load_dotenv
from src.helper import load_pdf,text_split,download_hugging_face_embeddings
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec


load_dotenv()  # take env variable from .env

Pinecone_API_KEY_=os.getenv("Pinecone_API_KEY")
Pinecone_API_env_= os.getenv("Pinecone_API_env")

####loading data and splitting data into chunks
extracted_data = load_pdf("./data")
text_chunks = text_split(extracted_data)
text_chunks1=[t.page_content for t in text_chunks]

#### downloading embeddin model and creates embedding or vectors for each chunks dim 384 x 7020
embeddings = download_hugging_face_embeddings()
#or
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings1 = model.encode(text_chunks1)

### creating index in pinecone if exist
pc = Pinecone(api_key=Pinecone_API_KEY_)
index_name = "med-chatbot2"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
index = pc.Index(index_name)

#### converting chunks meta data and embedding to req list format for uploading to pinecone vector db
vectors = []
i=1
for d, e in zip(text_chunks1, embeddings1):
    vectors.append({
        "id": str(i),
        "values": e,
        "metadata": {'text': d}
    })
    i=i+1

index.upsert(
    vectors=vectors,
    namespace="ns1",
    batch_size=1000
)




