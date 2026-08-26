from flask import Flask ,render_template,jsonify,request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from src.crag import build_crag_chain
import os

app=Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY=os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY




embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot" 
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)
docsearch = PineconeVectorStore(index=index, embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

chatModel = ChatGroq(model="llama-3.3-70b-versatile")

crag_chain = build_crag_chain(retriever, chatModel)


@app.route("/get",methods=["GET","POST"])
def chat():
    msg=request.form["msg"]
    input=msg
    print(input)
    response=crag_chain.invoke({"question":msg,"max_retries":2})
    print("Response:",response["generation"])
    return str(response["generation"])



if __name__=='__main__':
    app.run(host="0.0.0.0",port=8080,debug=True)
