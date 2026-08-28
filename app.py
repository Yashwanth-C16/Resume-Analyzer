import os
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")


st.title("Resume Analyzer")


#llm
llm=ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=groq_api_key
)

#user query
query=st.text_input("enter your query")


#prompt
prompt = ChatPromptTemplate.from_messages(
    [
    (
        "system",
        """You are a resume analyzer.
        Analyze the resume based only on the provided context.

        Resume:
        {context}"""
            ),
        ("user", "{query}")
    ])


#data ingestion
resume=st.file_uploader("add pdf",type=["pdf"])

button=st.button("analyze")

if resume:
    with open("resume.pdf", "wb") as f:
        f.write(resume.getbuffer())
    #pdf loader
    loader=PyPDFLoader("resume.pdf")
    texts=loader.load()

    #extract
    split_text=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    docs=split_text.split_documents(texts)

    #embeding
    embedding=OllamaEmbeddings(model="nomic-embed-text")

    #vectordb
    db=Chroma.from_documents(docs,embedding)

    if button:
        if query:
            #similarity search
            res=db.similarity_search(query,k=3)


            #output parser
            parser=StrOutputParser()


            context = "\n\n".join(doc.page_content for doc in res)

            chain=prompt|llm|parser

            ans = chain.invoke({
                    "query": query,
                    "context": context
                })

            st.success(ans)
        else:
            st.warning("enter query")
