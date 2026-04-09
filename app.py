# ============================================================
# GHANA RAG FINANCIAL ADVISOR BOT — STREAMLIT APPLICATION
# Project: Ghana Financial Sector Portfolio — Project 2
# Author: Evans Ataaya
# Date: April 2026
# ============================================================

import streamlit as st
import faiss
import pickle
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# --- Load Environment ---
load_dotenv()

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
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# --- Load Models and Data (cached for performance) ---
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_faiss_index():
    index = faiss.read_index('data/vectorstore/ghana_faiss.index')
    with open('data/vectorstore/chunk_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    return index, metadata

@st.cache_resource
def load_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


# --- RAG Functions ---
def search_knowledge_base(query, embedding_model, index, metadata,
                          top_k=5, domain_filter=None, body_filter=None):
    query_vector = embedding_model.encode([query])
    faiss.normalize_L2(query_vector)
    
    search_k = top_k * 5 if (domain_filter or body_filter) else top_k
    scores, indices = index.search(query_vector, min(search_k, index.ntotal))
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
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
            "similarity": float(score),
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


def ask_advisor(query, client, embedding_model, index, metadata,
                top_k=5, domain_filter=None, body_filter=None):
    
    results = search_knowledge_base(
        query, embedding_model, index, metadata,
        top_k=top_k, domain_filter=domain_filter,
        body_filter=body_filter
    )
    
    if not results:
        return {
            "response": "I could not find relevant information in the "
                       "Ghana financial regulatory documents to answer "
                       "your question.",
            "sources": [],
        }
    
    context_parts = []
    sources = []
    for i, r in enumerate(results):
        context_parts.append(
            f"[Source {i+1}: {r['title']} — {r['regulatory_body']}]\n"
            f"{r['text']}"
        )
        sources.append(r)
    
    context = "\n\n---\n\n".join(context_parts)
    
    user_prompt = f"""Based on the following context from Ghana's financial 
regulatory documents, answer the user's question accurately and cite your sources.

CONTEXT:
{context}

USER QUESTION: {query}

Provide a clear, comprehensive answer based ONLY on the context above."""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=1000,
        temperature=0.3,
    )
    
    return {
        "response": response.choices[0].message.content,
        "sources": sources,
        "tokens": response.usage.total_tokens,
    }


# --- Initialize ---
embedding_model = load_embedding_model()
index, chunk_metadata = load_faiss_index()
client = load_openai_client()

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 🇬🇭 Ghana Financial Advisor")
    st.markdown("---")
    
    st.markdown("#### Filter by Domain")
    domain_options = ["All Domains"] + sorted(set(
        m['domain'] for m in chunk_metadata))
    selected_domain = st.selectbox(
        "Knowledge Domain", domain_options, label_visibility="collapsed")
    
    st.markdown("#### Filter by Regulatory Body")
    body_options = ["All Bodies"] + sorted(set(
        m['regulatory_body'] for m in chunk_metadata))
    selected_body = st.selectbox(
        "Regulatory Body", body_options, label_visibility="collapsed")
    
    st.markdown("---")
    
    st.markdown("#### Example Questions")
    example_questions = [
        "How do I invest in Treasury bills?",
        "What is the minimum capital for banks?",
        "How does the pension system work?",
        "What are the rules for mobile money?",
        "What is the eCedi?",
        "How do I file a financial complaint?",
    ]
    
    for eq in example_questions:
        if st.button(eq, key=eq, use_container_width=True):
            st.session_state.pending_question = eq
    
    st.markdown("---")
    st.markdown("#### About")
    st.markdown(
        "**RAG Financial Advisor Bot**\n\n"
        "Built by Evans Ataaya\n"
        "MTech Data Science\n\n"
        "Knowledge base: 25 documents from "
        "Bank of Ghana, SEC, NIC, NPRA, "
        "Ministry of Finance\n\n"
        "Model: GPT-4o-mini + FAISS\n"
        "Embeddings: all-MiniLM-L6-v2"
    )

# --- Main Interface ---
st.markdown('<p class="main-header">Ghana Financial Advisor Bot</p>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">'
    'AI-powered financial guidance grounded in Ghana\'s regulatory framework'
    '</p>',
    unsafe_allow_html=True
)

# Disclaimer banner
st.markdown(
    '<div class="disclaimer">'
    '⚠️ <strong>Important:</strong> This tool provides information from '
    'official Ghanaian regulatory documents for educational purposes only. '
    'It does not constitute professional financial advice. Always consult '
    'a licensed financial advisor for specific guidance.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("")

# Check API key
if not client:
    st.error("OpenAI API key not found. Please add OPENAI_API_KEY to your .env file.")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📄 View Sources"):
                for src in message["sources"]:
                    st.markdown(
                        f'<div class="source-box">'
                        f'<strong>{src["title"]}</strong><br>'
                        f'{src["regulatory_body"]} | {src["domain"]}<br>'
                        f'Similarity: {src["similarity"]:.4f}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

# Handle pending question from sidebar
if "pending_question" in st.session_state:
    prompt = st.session_state.pending_question
    del st.session_state.pending_question
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching Ghana financial documents..."):
            domain_f = None if selected_domain == "All Domains" else selected_domain
            body_f = None if selected_body == "All Bodies" else selected_body
            
            result = ask_advisor(
                prompt, client, embedding_model, index,
                chunk_metadata, domain_filter=domain_f,
                body_filter=body_f
            )
        
        st.markdown(result["response"])
        
        if result["sources"]:
            with st.expander("📄 View Sources"):
                for src in result["sources"]:
                    st.markdown(
                        f'<div class="source-box">'
                        f'<strong>{src["title"]}</strong><br>'
                        f'{src["regulatory_body"]} | {src["domain"]}<br>'
                        f'Similarity: {src["similarity"]:.4f}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["response"],
        "sources": result["sources"]
    })
    st.rerun()

# Chat input
if prompt := st.chat_input("Ask about Ghana's financial regulations..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching Ghana financial documents..."):
            domain_f = None if selected_domain == "All Domains" else selected_domain
            body_f = None if selected_body == "All Bodies" else selected_body
            
            result = ask_advisor(
                prompt, client, embedding_model, index,
                chunk_metadata, domain_filter=domain_f,
                body_filter=body_f
            )
        
        st.markdown(result["response"])
        
        if result["sources"]:
            with st.expander("📄 View Sources"):
                for src in result["sources"]:
                    st.markdown(
                        f'<div class="source-box">'
                        f'<strong>{src["title"]}</strong><br>'
                        f'{src["regulatory_body"]} | {src["domain"]}<br>'
                        f'Similarity: {src["similarity"]:.4f}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["response"],
        "sources": result["sources"]
    })
    st.rerun()