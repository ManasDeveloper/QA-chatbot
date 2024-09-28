import os
from dotenv import load_dotenv

import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import OllamaEmbeddings


load_dotenv()

#LANGSMITH TRACKING
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A CHATBOT With OpenAI"


# prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant, please respond to the user queries"),
        ("user","Question : {question}")
    ]
)


def generate_response(question,api_key,llm,temperature,max_token):
    openai.api_key = api_key
    llm = ChatOpenAI(model=llm,api_key=api_key)
    output_parser = StrOutputParser()
    chain = prompt_template|llm|output_parser
    answer = chain.invoke({"question" : question})
    return answer


# streamlit application
st.title("Enhanced Q&A Chatbot with OpenAI")

# Sidebar for options
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter you OpenAI API Key",type="password")

## Dropdown to select various openai models
llm = st.sidebar.selectbox("Select an OpenAI Model",["gpt-4o","gpt-4-turbo","gpt-4"])

temperature = st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)

# Main Interface for user input
st.write("Ask any Question....")
question = st.text_input("You : ")

if question:
    response = generate_response(question=question,api_key=api_key,llm=llm,temperature=temperature,max_token=max_tokens)
    st.write(response)
else:
    st.write("Please provide the query...")
