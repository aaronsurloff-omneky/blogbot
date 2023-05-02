## Bring in deps
import streamlit as st
from langchain.llms import OpenAI
from langchain import SerpAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

openai_api_key = st.secrets["openai_apikey"]
serpapi_apikey = st.secrets["serpapi_apikey"]

## App Framework
st.title('Omneky Blog Bot')
prompt = st.text_input('Plug In Your Blog Topic Here')

## Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'],
    template = 'write me a blog title about {topic}'
)

blog_template = PromptTemplate(
    input_variables = ['title', 'google_research'],
    template = 'write me a 1500 character blog based on this title: {title} while leveraging this google research: {google_research}'
)

## Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
blog_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


## LLMS
llm = OpenAI(temperature=0.9, max_tokens = 1000, openai_api_key = openai_api_key)
tools = load_tools(["serpapi"], llm=llm, serpapi_api_key = serpapi_apikey)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

title_chain = LLMChain(llm=llm, prompt = title_template, verbose = True, output_key='title', memory=title_memory)
blog_chain = LLMChain(llm=llm, prompt = blog_template, verbose = True, output_key='blog', memory=blog_memory)

## show stuff to screen if there is a prompt
if prompt:
    title = title_chain.run(prompt)
    st.write(title)
    st.spinner('Writing Your Blog...')
    time.sleep(5)
    google_research = agent.run(title)
    blog = blog_chain.run(title=title,  google_research=google_research)
    
    st.write(blog)

    with st.expander('Title History'):
        st.info(title_memory.buffer)
    
    with st.expander('Blog History'):
        st.info(blog_memory.buffer)

    with st.expander('Google Research'):
        st.info(google_research)
