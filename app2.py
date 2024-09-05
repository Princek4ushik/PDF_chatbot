#PDF_chatbot
#imports
import streamlit as st
import pickle
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from transformers import pipeline
#pipeline was added as a last ditch effort to try another approach
#os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
#this peice is commented out because of the use of ,env file
st.header("Querry your PDF")
load_dotenv()
    #if statement used to prevent error from accuring if no file is uploaded
pdf= st.file_uploader("Upload PDF", type="pdf")
if pdf is not None:
    st.write(pdf.name)
    pdf_reader= PdfReader(pdf)
    #empty object text is used to read and then append all the text in pdf to it
    text=""
    for page in pdf_reader.pages:
        text +=page.extract_text()
    #splitting the data into chunks for better readability and performance
    #overlap is important to not miss context/content         
    text_splitter= RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,length_function=len
    )
    chunks= text_splitter.split_text(text=text)
    store_name= pdf.name[:-4]

#money saver code
    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl","rb") as f:
            VectorStore=pickle.load(f)
    else:
        embeddings= HuggingFaceEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"{store_name}.pkl","wb") as f: pickle.dump(VectorStore, f)
#
#
    query= st.text_input("Querry your uploaded PDF file:")
    st.write(query)
    if query:
        docs= VectorStore.similarity_search(query=query, k=3)
        #k defines the limit of relevent docs/context etc
        #llm=HuggingFaceHub(repo_id="naver-clova-ix/donut-base-finetuned-docvqa",) 
        #better configuration can be figured out with more time to read and experiment with the model
        #once a better configuration is found llm=llm should be used in the next line making all this functional
        pipe = pipeline("document-question-answering", model="naver-clova-ix/donut-base-finetuned-docvqa")
        chain= load_qa_chain(llm=HuggingFaceHub(), chain_type="stuff")
        #four different chain types are available
        response= chain.run(input_documnets=docs, question=query)
        st.write(response)
#