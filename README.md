# Nexus Chatbot 🧠✨
An intelligent AI agent that acts as a central hub for information, dynamically connecting user-uploaded documents, the live web, and its own internal knowledge to provide accurate, context-aware answers.

[🚀Access the Live Demo Here!](https://nexus-chatbot24.streamlit.app/)

## 📖 The Problem
In a world filled with information, valuable knowledge is often trapped in dense documents or is so recent that standard AI models are unaware of it. Getting a specific, trustworthy answer requires sifting through lengthy reports or searching the web for hours.

Nexus Chatbot was built to solve this. It's an AI research assistant that can instantly become an expert on any topic by reading documents you provide, browsing the live internet for current events, and using its own reasoning to synthesize the best possible answer.

## ✨ Key Features
The chatbot integrates multiple advanced features into a single, intuitive interface, creating a versatile and powerful information tool.

### 🧠 Intelligent Agent Router
The chatbot’s core is a smart agent that analyzes each query to determine the best source of information. It intelligently chooses between:

Searching the uploaded document.

Browsing the live web.

Using its own general knowledge.

### 📂 Private Document Analysis (RAG)
Upload your own PDF or TXT files to instantly turn the chatbot into a temporary expert on that specific content. Perfect for deep, contextual question-answering on research papers, legal documents, or technical manuals.

### 🌐 Real-Time Web Search
For questions about current events, news, or any topic not covered in the document, the agent can access the live internet via the Tavily search tool to provide up-to-the-minute, verifiable answers.

### ⚙️ Flexible Multi-Provider Backend
The system is not locked into one provider. The UI allows you to seamlessly switch between:

Large Language Models: Use the ultra-fast Groq (Llama 3.1) for speed or other models for different tasks.

Embedding Models: Choose between powerful API-based models or a completely private, local embedding model for 100% data security.

## 📝 Configurable Response Style
A simple UI toggle lets you control the verbosity of the AI's answers. You can request either a Concise, one-paragraph summary or a Detailed, in-depth explanation for any response.

## 🛠️ Tech Stack
Framework: LangChain, Streamlit

LLM Providers: Groq, OpenAI, Google Gemini

Vector Store: FAISS (for in-memory similarity search)

Web Search: Tavily AI

Language: Python

## ⚙️ How It Works: The Agent's Thought Process
The Nexus Chatbot operates as an intelligent agent, following a logical process for every query to ensure the highest quality response.

Query Analysis: When a query is submitted, it's first sent to a lightweight, high-speed "Router" model.

Intelligent Routing: This router analyzes the user's intent and makes a critical decision: is the answer likely to be in the uploaded private document, or does it require real-time information from the internet?

Tool Execution: Based on the router's decision, the appropriate tool is activated:

Document Search (RAG): If the query is about the uploaded file, a semantic search is performed against the FAISS vector store to find the most relevant text chunks.

Web Search: If the query is about a current event, the Tavily AI search tool is used to browse the live internet for up-to-the-minute information.

Context Augmentation: The retrieved information (from either the document or the web) is then combined with the original query and the system instructions.

Final Generation: This complete, context-rich prompt is sent to the main Large Language Model (e.g., Groq's Llama 3.1), which synthesizes a final, accurate, and human-readable answer.

## 🚀 Future Improvements
This project serves as a strong foundation for an even more capable AI agent. Future development could include:

Conversational Memory: Implementing a more advanced memory system to allow for more natural follow-up questions and context retention across longer conversations.

Support for More File Types: Expanding the document loader to handle .docx, .csv, and even images for multimodal analysis.

Additional Tools: Integrating more tools for the agent to use, such as a calculator for mathematical queries or a calendar API for scheduling questions.

Enhanced UI/UX: Adding features like clickable source links directly in the chat, allowing users to verify information from the web or the source document instantly.

Web Search: Tavily AI

Language: Python
