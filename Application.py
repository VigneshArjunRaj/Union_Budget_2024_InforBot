from concurrent.futures import ThreadPoolExecutor
import streamlit as st
from LLMControllerOllama import init_agent_executor
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
import streamlit as st

st_callback = StreamlitCallbackHandler(st.container())

def process_query(agent_executor, query):
    return agent_executor.invoke({'input':query})

@st.cache_resource
def get_agentexecutor_and_threadexecutor():
    executor = ThreadPoolExecutor(max_workers=5)
    llm_agent_executor = init_agent_executor(model_name="OpenAI")
    return executor,llm_agent_executor

st.title("Indian Budget 2024-2025 Analysis")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
# query = st.chat_message("Enter your question about the Indian Budget:")

executor, agent_executor = get_agentexecutor_and_threadexecutor()
if prompt := st.chat_input(placeholder="Enter your question about the Indian Budget:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor.invoke(
            {"input": prompt,"chat_history":st.session_state.messages},
            {"callbacks": [st_callback]},

        )
        st.write(response["output"])
    st.session_state.messages.append({"role": "assistant", "content": response["output"]})

# if query:
#     with st.spinner("Analyzing the budget..."):
#         # future = executor.submit(process_query, agent_executor, query)
#         # result = future.result()
#         st.write("random check")




