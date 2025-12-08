from http import client
import streamlit as st
from openai import api_key
from google import genai
from openai import OpenAI
import os
from datetime import datetime
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.chat_engine import CondensePlusContextChatEngine

from src.model_loader import get_embedding_model, initialise_llm, initialise_llm
from src.engine import get_chat_engine

# ------ BOT INITIALISATION ------ #

@st.cache_resource #bot must be cached
def initialise_bot() -> CondensePlusContextChatEngine:
    # Get LLM and embedding model
    llm: GoogleGenAI = initialise_llm()
    embed_model: HuggingFaceEmbedding = get_embedding_model()
    # Create RAG chat bot
    chat_engine: CondensePlusContextChatEngine = get_chat_engine(llm, embed_model)
    return chat_engine

bot: CondensePlusContextChatEngine = initialise_bot()



# ============================
# Page Config
# ============================
st.set_page_config(
    page_title="Tattoo Aftercare Chatbot",
    page_icon="💭",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================
# Modern Dark CSS + ANIMATED GLOW HEADER
# ============================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Megrim&family=Vollkorn+SC:wght@400;600;700&display=swap');

    /* Root */
    .stApp {background: #0a0a0a; color: #e0e0e0; font-family: 'Vollkorn SC', serif;}

    /* FIXED HEADER */
    .fixed-header {
        position: fixed;
        top: 0; left: 0; right: 0;
        background: linear-gradient(180deg, #0f0f0f 0%, #0a0a0a 100%);
        padding: 2.5rem 1rem 1.8rem;
        text-align: center;
        border-bottom: 2px solid transparent;
        z-index: 999;
        backdrop-filter: blur(12px);
        animation: headerPulse 6s infinite ease-in-out;
    }

    /* Pulsing border glow */
    @keyframes headerPulse {
        0%, 100%   { border-bottom-color: rgba(0, 230, 195, 0.4); box-shadow: 0 4px 30px rgba(0, 230, 195, 0.1); }
        50%        { border-bottom-color: #00e6c3;       box-shadow: 0 4px 50px rgba(0, 230, 195, 0.4); }
    }

    .main-title {
        font-family: 'Megrim', cursive;
        font-size: 5.8rem;
        margin: 0;
        color: #00e6c3;
        letter-spacing: 4px;
        background: linear-gradient(90deg, #00e6c3, #00ffcc, #00e6c3);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: titleGlow 4s infinite ease-in-out;
    }

    /* Animated neon text glow */
    @keyframes titleGlow {
        0%, 100%   { text-shadow: 0 0 20px rgba(0, 230, 195, 0.6); }
        50%        { text-shadow: 0 0 40px rgba(0, 230, 195, 0.9), 0 0 60px rgba(0, 255, 204, 0.4); }
    }

    .subtitle {
        font-family: 'Vollkorn SC', serif;
        font-size: 1.1rem;
        color: #00e6c3;
        letter-spacing: 7px;
        margin-top: 10px;
        opacity: 0.85;
        text-transform: uppercase;
        animation: subtitleFade 6s infinite ease-in-out;
    }

    @keyframes subtitleFade {
        0%, 100%   { opacity: 0.7; }
        50%        { opacity: 1.0; }
    }

    /* Scrollable chat area */
    .chat-container {
        margin-top: 180px;
        padding: 1rem 1.5rem 140px;
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
    }

    /* Bubbles – unchanged but kept teal */
    .user-bubble, .assistant-bubble {
        backdrop-filter: blur(12px);
        border: 1px solid #00e6c3;
        border-radius: 18px;
        padding: 1rem 1.3rem;
        margin: 1.2rem 0;
        max-width: 82%;
        box-shadow: 0 8px 25px rgba(0, 230, 195, 0.18);
    }
    .user-bubble      { background: rgba(15,15,15,0.9); border-radius: 18px 18px 4px 18px; align-self: flex-end; }
    .assistant-bubble { background: rgba(10,10,10,0.95); border-radius: 18px 18px 18px 4px; align-self: flex-start; color: #ccfff9; }

    /* Input glow */
    .stTextInput > div > div > input {
        background: #0f0f0f !important;
        color: #ccfff9 !important;
        border: 1px solid #004d40 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        transition: all 0.3s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #00e6c3 !important;
        box-shadow: 0 0 30px rgba(0, 230, 195, 0.6) !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {width: 8px;}
    ::-webkit-scrollbar-track {background: #0a0a0a;}
    ::-webkit-scrollbar-thumb {background: #00e6c3; border-radius: 4px;}
    ::-webkit-scrollbar-thumb:hover {background: #00ffcc;}

    /* Hide Streamlit UI */
    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# AUTO-SCROLL (clean & reliable)
js = '''
<script>
    const observer = new MutationObserver(() => {
        window.parent.document.querySelector(".main")?.scrollTo(0, document.body.scrollHeight);
    });
    observer.observe(document.body, { childList: true, subtree: true });
    // Initial scroll
    window.parent.document.querySelector(".main")?.scrollTo(0, document.body.scrollHeight);
</script>
'''
st.markdown(js, unsafe_allow_html=True)

# ============================
# Chat Interface
# ============================
# --- Displaying Chat History ---
# Initialise chat history if not already existing
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --- CHATBOT ---
# React to user input
if user_message := st.chat_input("💬"):
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_message)
    # Add to chat history
    st.session_state.messages.append({"role": "user", "content": user_message})
    
    # -- BOT RESPONSE --
    with st.chat_message("assistant"):

        # Have the spinner only for the retrieval
        with st.spinner("... 🤖 ..."):

            # .stream_chat() make it print bit by bit instead of waiting for the full block
            streaming_response = bot.stream_chat(user_message)

        response = st.write_stream(streaming_response.response_gen)

    # Add full bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})


# ============================
# Footer
# ============================
st.markdown("""
<div style="text-align:center; margin-top: 50px; color: #666; font-size: 0.9rem;">
    <p> 🩵 Heal. Well. 🩵 </p>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ============================
# Animated Fixed Header
# ============================
st.markdown("""
<div class="fixed-header">
    <h1 class="main-title">TAT CHAT</h1>
    <div class="subtitle">Expert chatbot for tattoo healing and infection risks</div>
</div>
""", unsafe_allow_html=True)

