"""运行配置：优先读取 Streamlit Secrets，本地运行时回退到系统环境变量。"""

import os

import streamlit as st

from .log import get_logger

logger = get_logger("config")


def get_secret(name, default=""):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception as e:
        logger.debug("secrets_unavailable key=%s error=%s", name, e)
    return os.getenv(name, default)


def clean_ss_key(raw):
    """去掉粘贴污染（引号、换行、空格）。带污染的 Key 会被服务端一律判 403。"""
    return str(raw or "").replace("\r", "").replace("\n", "").strip().strip("\"'").strip()


USER_API_KEY = get_secret("DEEPSEEK_API_KEY")
SS_API_KEY = clean_ss_key(get_secret("SS_API_KEY"))

PINECONE_API_KEY = get_secret("PINECONE_API_KEY")
PINECONE_INDEX_NAME = get_secret("PINECONE_INDEX_NAME", "arxiv-papers")
