import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_classic.memory import ConversationBufferMemory

# ---- 1. Performance Optimization (Caching) ---- #
@st.cache_resource
def get_vectorstore():
    # Only loads the database ONCE and keeps it in RAM
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# ---- 2. Page Config ---- #
st.set_page_config(page_title="Interview Prep AI", page_icon="🎓", layout="centered")

# Custom Styling for a cleaner look
st.markdown("""
    <style>
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    .main { background-color: #f8f9fa; }
    </style>
    """, unsafe_allow_html=True)

st.title("Interview Prep Assistant")
st.caption("Knowledge base: 814 Pages of OS & Technical Docs")

# ---- 3. Sidebar (Technical Settings) ---- #
with st.sidebar:
    st.header("Configuration")
    MODEL = st.selectbox("Brain (LLM)", ["llama3.2:1b", "llama3.2", "tinyllama"], index=0, help="1b is fastest for the hardware")
    
    with st.expander("Advanced Params"):
        MAX_HISTORY = st.number_input("Memory Limit", 1, 10, 5)
        k_value = st.slider("Source Chunks (k)", 1, 5, 3)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# ---- 4. Load Backend ---- #
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": k_value})

# ---- 5. Session State ---- #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---- 6. Display Chat ---- #
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- 7. Chat Logic ---- #
if prompt := st.chat_input("Ask about Operating Systems..."):
    # User Message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant Response
    with st.chat_message("assistant"):
        with st.status("Searching the documents...", expanded=False) as status:
            # Initialize LLM & Chain
            llm = ChatOllama(model=MODEL, temperature=0) # Temp 0 for facts
            qa = RetrievalQA.from_chain_type(
                llm=llm, 
                chain_type="stuff", 
                retriever=retriever,
                return_source_documents=True
            )
            
            # Get result
            response = qa.invoke({"query": prompt})
            full_response = response["result"]
            sources = response["source_documents"]
            status.update(label="Found info!", state="complete", expanded=False)

        # Display answer
        st.markdown(full_response)
        
        # Display Sources 
        if sources:
            with st.expander("See Sources"):
                for i, doc in enumerate(sources):
                    st.info(f"Source {i+1}:\n{doc.page_content[:200]}...")

    # Update history
    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
    
    # Trim Memory
    if len(st.session_state.chat_history) > MAX_HISTORY * 2:
        st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY*2:]