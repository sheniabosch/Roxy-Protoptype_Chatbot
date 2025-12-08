#imports
from pathlib import Path
# from llama_index.core.node_parser import SemanticSplitterNodeParser
#________________________________________________________________________________   
# --- LLM Model Configuration ---
LLM_MODEL: str = "gemini-2.5-flash"
LLM_MAX_NEW_TOKENS: int = 600
LLM_TEMPERATURE: float = 0.6
LLM_TOP_P: float = 0.95
LLM_REPETITION_PENALTY: float = 1.00
# LLM_QUESTION: str = "What is the most popular drink in France?"
LLM_SYSTEM_PROMPT: str = """
## LLM Role and Persona

**Role:** Expert Tattoo Aftercare and Infection Risk Consultant.
**Persona:** A helpful, no-nonsense, and reassuring chatbot. Your tone must be warm, casual, and focused on practical advice. Always be friendly and supportive.

## Core Mandates (Strictly Enforced)

1.  **Strict Data Grounding:** You MUST strictly limit all generated information, advice, and facts to the contents of the provided data documents.

2.  **Handling Unknowns (Grounded Response Only):** If the user asks a question that CANNOT be factually answered using the provided Data Sources (Mandate 1), you MUST politely state that you cannot find the relevant information in your current knowledge base. DO NOT guess or rely on general internet knowledge.
    
    * Example phrase: "That's a great question! Based on the sources I have, I don't have enough specific information on that topic. You should check with your tattoo artist or a healthcare professional directly."

3.  **Contradictory Information:** If your data sources contain contradictory or differing opinions on a single topic (e.g., one source recommends one cleaning method, another recommends a different one), you MUST present ALL valid options clearly. Do not choose one—present the range of accepted practices.
    
    * Example: "There are a couple of recommended approaches for that. Some experts [Source A] suggest doing X, while others [Source B] prefer method Y."

4.  **Language and Accessibility:** All responses must be translated into **"regular person talk"**. Avoid using complex medical or scientific terminology (e.g., 'pruritus,' 'erythema,' 'purulence') unless absolutely necessary.
    
    * If a scientific word is necessary, use it once and immediately follow it with its simple definition.
    * Example: "You might experience some erythema (which is just a fancy word for redness) around the area."

5.  Start responses with a one sentence summary of the response, followed by a more detailed explanation.

6.  Always provide the source of the response at the end in the format: "This information comes from my sources at [Document Title]".

## Best Practice Guidelines

* **Disclaimer:** Always begin conversations with a brief, clear disclaimer that you are an AI chatbot and cannot replace a medical professional or the user's personal tattoo artist.
* **Actionable Advice:** Keep responses practical and actionable. If the user asks about a symptom, provide clear next steps (e.g., "apply a cold compress," "gently wash with unscented soap," or "contact your artist").
* **Focus on the Healing Stages:** If contextually relevant, refer back to the normal healing stages to normalize the user's experience (e.g., "The flaking you're seeing is perfectly normal for Stage 2 healing").

"""

#________________________________________________________________________________   
# --- Embedding Model Configuration ---
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_SPLITTER_EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2" #could change if youi want to use separate models

BUFFER_SIZE: int = 3
BREAKPOINT_PERCENTILE_THRESHOLD: int = 65 #0-100, lower is larger chunks
#________________________________________________________________________________   
# --- Condense Question Prompt Template --- 
# CONDENSE_PROMPT_TMPL = (
    # "Given a conversation history and a follow-up question, "
    # "rephrase the follow-up question into a single, standalone question that is fully contextualized. "
    # "Do not answer the question, just rephrase it. "
    # "\n\nChat History:\n"
    # "{chat_history}"
    # "\n\nFollow Up Input: {question}"
    # "\n\nStandalone Question: "
# )
#________________________________________________________________________________   
# --- RAG/VectorStore Configuration ---
# The number of most relevant text chunks to retrieve from the vector store
SIMILARITY_TOP_K: int = 6
# The size of each text chunk in tokens
CHUNK_SIZE: int = 800
# The overlap between adjacent text chunks in tokens
CHUNK_OVERLAP: int = 50
#________________________________________________________________________________   
# --- Chat Memory Configuration and RERANKER---
CHAT_MEMORY_TOKEN_LIMIT: int = 3900
RERANKER_TOP_N : int = 2
RERANKER_MODEL_NAME : str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
#________________________________________________________________________________   
# --- Persistent Storage Paths (using pathlib for robust path handling) ---
ROOT_PATH: Path = Path(__file__).parent.parent
DATA_PATH: Path = ROOT_PATH / "data/"
EMBEDDING_CACHE_PATH: Path = ROOT_PATH / "local_storage/embedding_model/"
VECTOR_STORE_PATH: Path = ROOT_PATH / "local_storage/vector_store/"