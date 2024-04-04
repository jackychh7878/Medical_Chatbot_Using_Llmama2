from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from src.prompt import *
import os
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()

# Pinecone can read the pinecone_api_key once loaded, so even you commented it still work
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

embeddings = download_hugging_face_embeddings()

# Initilizing the Pinecone
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)

# Testing the similarity search
# query = "What are Allergies"
# docs = docsearch.similarity_search(query, k=3)
# print("Result", docs)

# Create the prompt template
PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens': 512, 'temperature': 0.8})

qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
                                 return_source_documents=True,
                                 chain_type_kwargs=chain_type_kwargs
                                 )
# user_input=input(f"Input Prompt:")
# result=qa({"query": user_input})
# print("Response : ", result["result"])


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)

