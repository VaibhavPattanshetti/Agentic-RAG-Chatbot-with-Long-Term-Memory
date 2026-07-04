import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import uuid

from langgraph_tool_backend import chatbot, retrive_all_threads, ingest_pdf

# ************************************** Utility Functions ***********************************************
def generate_thread_id():
    thread_id = str(uuid.uuid4())
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    add_thread(thread_id)
    st.session_state['thread_id'] = thread_id
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    
    state = chatbot.get_state(
    config={'configurable': {'thread_id': thread_id}}
    )

    return state.values.get('messages', [])



# ************************************** Session Setup ***********************************************

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrive_all_threads()

add_thread(st.session_state['thread_id'])  # Ensure the (1st)current thread is in the list of threads

if 'ingested_docs' not in st.session_state:
    st.session_state['ingested_docs'] = {}



thread_key = str(st.session_state['thread_id'])
thread_docs = st.session_state['ingested_docs'].setdefault(thread_key, {})

# *************************************** Sidebar UI ***********************************************

st.sidebar.title("LangGraph Chatbot")
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

# ---- PDF status card ----
if thread_docs:
    active_file = list(thread_docs.keys())[-1]
    st.sidebar.markdown("**📄 Active document**")
    st.sidebar.write(active_file)
else:
    st.sidebar.caption("No document uploaded yet for this chat.")

st.sidebar.divider()

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.caption(f"'{uploaded_pdf.name}' is already active.")
    else:
        with st.spinner("Reading and preparing your document..."):
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary

        st.sidebar.success(f"'{uploaded_pdf.name}' is ready to use.")
        st.rerun()

st.sidebar.header("My Conversations")
for thread_id in st.session_state['chat_threads'][::-1]: 
    if st.sidebar.button(thread_id):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                temp_messages.append({'role': 'user', 'content': msg.content})
            elif isinstance(msg, AIMessage) and msg.content:
                temp_messages.append({'role': 'assistant', 'content': msg.content})

        st.session_state['message_history'] = temp_messages


# *************************************** Main UI ***********************************************

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

#{'role': 'user', 'content': 'Hi'}
#{'role': 'assistant', 'content': 'Hi=ello'}

user_input = st.chat_input('Type here')



if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)


    # config={
    #     'configurable': {
    #         'thread_id': st.session_state['thread_id']
    #     }
    config = {
        "configurable": {"thread_id": st.session_state["thread_id"], "user_id": "vaibhav"},
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_turn",
    }


     # Assistant streaming block
    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            ):
                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )
                if metadata.get("langgraph_node") != "chat_node":
                    continue
                # Stream ONLY assistant messages
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )
    

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})