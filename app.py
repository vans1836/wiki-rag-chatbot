import gradio as gr
from langchain_community.document_loaders import WikipediaLoader
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os

# Setting up my Cohere API Key
os.environ["COHERE_API_KEY"] = "ZzVFCGrAWfsUKkcuj2JnIM9H4X00s8u83i08FXKq"  

# Loading and preparing RAG pipeline once
def build_rag_chain(question):
    loader = WikipediaLoader(query=question, load_max_docs=3)
    docs = loader.load()
    embeddings = CohereEmbeddings(model="embed-english-v3.0")
    vector_store = FAISS.from_documents(docs, embeddings)
    qa = RetrievalQA.from_chain_type(llm=ChatCohere(), retriever=vector_store.as_retriever())
    response = qa.invoke({"query": question})
    return response

# Creating the  interface
iface = gr.Interface(
    fn=build_rag_chain,
    inputs=gr.Textbox(lines=2, placeholder="Ask me anything about the world..."),
    outputs="text",
    title="🌐 Wiki-Powered RAG Chatbot",
    description="Powered by Cohere + Wikipedia + FAISS + LangChain"
)

iface.launch()
