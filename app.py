from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain_groq import ChatGroq
import streamlit as st
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun, DuckDuckGoSearchResults
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain.agents import initialize_agent, AgentType

st.title("LangChain - Search with langchain")

api_key = st.text_input("Write GROQ API Key", type="password")

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search = DuckDuckGoSearchResults(name='Search')

if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "assistant", "content": "Hi, I am a chatbot who can search web. How can I assist you?"}
    ]
    
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message('user').write(prompt)

    llm = ChatGroq(model="Llama3-8b-8192", groq_api_key = api_key, streaming=True)
    tools = [wiki,arxiv, search]

    agent = initialize_agent(llm=llm, tools=tools, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors = True)

    if st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})