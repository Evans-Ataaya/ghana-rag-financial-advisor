# ============================================================
# GHANA RAG FINANCIAL ADVISOR BOT — STREAMLIT APPLICATION
# Project: Ghana Financial Sector Portfolio — Project 2
# Author: Evans Ataaya
# Date: April 2026
# ============================================================

import streamlit as st
import numpy as np
import os
import json
import pickle
from openai import OpenAI

# --- Page Configuration ---
st.set_page_config(
    page_title="Ghana Financial Advisor",
    page_icon="🇬🇭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 28px;
        font-weight: bold;
        color: #1B4F72;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 14px;
        color: #666;
        margin-bottom: 20px;
    }
    .source-box {
        background-color: #f0f7fb;
        border-left: 4px solid #1B4F72;
        padding: 10px 15px;
        margin: 5px 0;
        border-radius: 0 5px 5px 0;
        font-size: 13px;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        padding: 10px 15px;
        border-radius: 5px;
        font-size: 12px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


# --- Load Models and Data ---
@st.cache_resource
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_knowledge_base():
    """Load chunk metadata and build numpy search index."""
    with open('data/vectorstore/chunk_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    # Re-embed all chunks on startup (avoids FAISS compatibility issues)
    model = load_embedding_model()
    texts = [m['text'] for m in metadata]
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)
    # Normalise for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    return metadata, embeddings

@st.cache_resource
def load_openai_client():
    # Try Streamlit secrets first, then environment variable
    api_key = None
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


# --- Search Function (numpy-based, no FAISS needed) ---
def search_knowledge_base(query, embedding_model, metadata, embeddings,
                          top_k=5, domain_filter=None, body_filter=None):
    query_vector = embedding_model.encode([query])
    query_vector = query_vector / np.linalg.norm(query_vector)
    
    # Compute cosine similarity
    similarities = np.dot(embeddings, query_vector.T).flatten()
    
    # Get top results
    top_indices = np.argsort(similarities)[::-1]
    
    results = []
    for idx in top_indices:
        meta = metadata[idx]
        if domain_filter and meta['domain'] != domain_filter:
            continue
        if body_filter and meta['regulatory_body'] != body_filter:
            continue
        results.append({
            "chunk_id": meta['chunk_id'],
            "doc_id": meta['doc_id'],
            "title": meta['title'],
            "domain": meta['domain'],
            "doc_type": meta['doc_type'],
            "regulatory_body": meta['regulatory_body'],
            "text": meta['text'],
            "similarity": float(similarities[idx]),
        })
        if len(results) >= top_k:
            break
    return results


SYSTEM_PROMPT = """You are the Ghana Financial Advisor Bot, an AI-powered financial 
information assistant specialising in Ghana's financial sector. Your knowledge comes 
exclusively from official Ghanaian regulatory documents including legislation, 
directives, and guidelines from the Bank of Ghana, the Securities and Exchange 
Commission (SEC), the National Insurance Commission (NIC), the National Pensions 
Regulatory Authority (NPRA), and the Ministry of Finance.

RULES YOU MUST FOLLOW:
1. ONLY use information from the provided context documents to answer questions.
2. ALWAYS cite your sources by mentioning the document title and regulatory body.
3. Present information clearly and accessibly for non-specialist users.
4. Use Ghana-specific terminology (cedis, GHS, Bank of Ghana, etc.).
5. If the context does not contain relevant information, say so honestly.
6. ALWAYS end with: "Disclaimer: This information is for educational purposes only 
   and does not constitute professional financial advice. Please consult a licensed 
   financial advisor or the relevant regulatory body for specific guidance."
"""


def ask_advisor(query, client, embedding_model, metadata, embeddings, top_k=5, domain_filter=None, body_filter=None):
