from langchain.chat_models import ChatOpenAI
from langchain.schema import(
    SystemMessage,
    HumanMessage,
    AIMessage
)

import streamlit as st
from streamlit_chat import message


from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

st.set_page_config(
    page_title="Your Custom Assistant",
    page_icon=' 🤖 ' 
)
st.subheader('You Custom ChatGPT 🤖')

chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5)

if 'messages' not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    system_message = st.text_input(label='System role')
    user_prompt = st.text_input(label='Send Message')

    if system_message:
        if not any(isinstance(x, SystemMessage) for x in st.session_state.messages):
            st.session_state.messages.append(
                SystemMessage(content=system_message)
            )
        st.write(st.session_state.messages)
    
    if user_prompt:
        st.session_state.messages.append(
            HumanMessage(content=user_prompt)
        )

        with st.spinner('Working on your request ....'):
            response = chat(st.session_state.messages)

    
        st.session_state.messages.append(AIMessage(content=response.content))

st.session_state.messages