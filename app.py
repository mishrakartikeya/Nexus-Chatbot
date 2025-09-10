import streamlit as st
import os
import sys
import logging
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from models.llm import load_llm
from utils.rag_utils import load_and_index_docs, retrieve_relevant_chunks, perform_web_search

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.llm import get_chatgroq_model
from utils.rag_utils import load_and_index_docs, retrieve_relevant_chunks


def get_chat_response(chat_model, messages, system_prompt, retriever=None, user_input=None, embeddings_provider="local", style="Detailed"):
    if not user_input:
        return "Please enter a message.", []

    try:
        # --- INTELLIGENT ROUTER ---
        router_prompt = f"""
            Based on the user's latest question, should I use a real-time web search tool or search the provided document context?
            The user's question is: "{user_input}"
            If the question is about recent events, current affairs, stock prices, weather, or topics not likely to be in a static document, respond with only the word 'web_search'.
            Otherwise, respond with only the word 'document_search'.
        """
        router_model = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
        decision = router_model.invoke(router_prompt).content.strip().lower()

        context = ""
        source_docs = []
        footer = "\n\n💻 From LLM" 

        if "web_search" in decision:
            print(f"INFO: Router chose [web_search] for query: '{user_input}'")
            context = perform_web_search(user_input)
            footer = "\n\n🌐 From Web Search"
        else:
            print(f"INFO: Router chose [document_search] for query: '{user_input}'")
            if retriever:
                index, chunks = retriever
                docs = retrieve_relevant_chunks(user_input, index, chunks, provider=embeddings_provider)
                if docs:
                    source_docs = docs
                    context = "\n\n---\n\n".join(docs)
                    footer = "\n\n📂 From uploaded document"

        # --- DYNAMIC PROMPT ENGINEERING ---
        final_system_prompt = system_prompt
        if context:
            final_system_prompt += f"\n\nUse the following context to answer the user's question:\n{context}"

        if style == "Concise":
            final_system_prompt += "\n\nImportant: Provide a concise, one-paragraph summary as your answer. Be brief and to the point."
        else: 
            final_system_prompt += "\n\nImportant: Provide a detailed, in-depth response. Use bullet points or lists if it helps to structure the information."

        # --- RESPONSE GENERATION ---
        formatted_messages = [SystemMessage(content=final_system_prompt)]
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                formatted_messages.append(HumanMessage(content=content))
            else:
                cleaned_content = content.split("\n\n")[0]
                formatted_messages.append(AIMessage(content=cleaned_content))
        
        response = chat_model.invoke(formatted_messages)
        answer = response.content + footer

        return answer, source_docs

    except Exception as e:
        logging.error(f"Error in get_chat_response: {str(e)}")
        return f"I'm sorry, an error occurred: {str(e)}", []


def chat_page():
    """Main chat interface page"""
    st.title("🤖 AI ChatBot")

    st.sidebar.divider()
    st.sidebar.header("📝 Response Style")
    response_style = st.sidebar.radio(
        "Select the level of detail:",
        ("Concise", "Detailed"),
        index=1  # Default to "Detailed"
    )

    st.sidebar.header("⚙️ Model Configuration")
    llm_provider_choice = st.sidebar.selectbox(
    "Choose your Language Model:",
    ("OpenAI", "Groq", "Gemini"))

    embeddings_provider_choice = st.sidebar.selectbox(
    "Choose your Document Embeddings:",
    ("OpenAI", "Gemini", "Local"))

    chat_model = load_llm(provider=llm_provider_choice.lower())
    chosen_embeddings_provider = embeddings_provider_choice.lower()
    st.sidebar.info(f"Using **{llm_provider_choice}** for chat and **{embeddings_provider_choice}** for documents.")

    st.sidebar.header("📂 Upload Knowledge Base")
    uploaded_files = st.sidebar.file_uploader(
        "Upload cultural heritage documents (PDF/TXT)", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    retriever = None
    if uploaded_files:
        index, chunks = load_and_index_docs(uploaded_files,provider=chosen_embeddings_provider)
        if index and chunks:
            retriever = (index, chunks)
            st.sidebar.success("✅ Documents indexed and ready!")
        else:
            st.sidebar.error("⚠️ Failed to index documents.")

    # Default system prompt
    system_prompt = "You are a helpful AI assistant specializing in cultural heritage."

    # Load chat model
    chat_model = get_chatgroq_model()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Getting response..."):
                response_text , source_docs = get_chat_response(chat_model, 
                st.session_state.messages, 
                system_prompt, 
                retriever, 
                prompt,
                embeddings_provider=chosen_embeddings_provider,
                style=response_style)
                st.markdown(response_text)

        st.session_state.messages.append({"role": "assistant", "content": response_text})


def main():
    st.set_page_config(
        page_title="LangChain Multi-Provider ChatBot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Go to:",
            ["Chat", "Instructions"],
            index=0
        )

        if page == "Chat":
            st.divider()
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    if page == "Instructions":
        instructions_page()
    if page == "Chat":
        chat_page()


if __name__ == "__main__":
    main()
