"""Embedding 与 Cross-Encoder 模型（进程级缓存，避免重复加载）。"""

import streamlit as st


@st.cache_resource
def get_embeddings_model():
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def get_cross_encoder():
    from sentence_transformers import CrossEncoder

    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
