#Final Working File
import asyncio
import uuid
import json
import re
import io
import os
import time
import tempfile
from datetime import datetime

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

import google.generativeai as genai
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.vectordb.chroma import ChromaDb
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

COLLECTION_NAME = "deepseek_rag"
EMBEDDING_MODEL = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
FAST_MODEL      = "gemini-2.5-flash"

st.set_page_config(
    page_title="Gemini RAG Agent",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --navy-darkest:  #020b18;
    --navy-dark:     #071428;
    --navy-mid:      #0d2247;
    --navy-light:    #153370;
    --navy-accent:   #1e4db7;
    --cyan-glow:     #00d4ff;
    --cyan-soft:     #7dd3fc;
    --gold:          #fbbf24;
    --text-primary:  #e8f4fd;
    --text-secondary:#94b8d4;
    --text-muted:    #4d7a9e;
    --card-bg:       rgba(13,34,71,0.85);
    --border:        rgba(0,212,255,0.18);
    --shadow:        0 8px 32px rgba(0,212,255,0.08);
}

.stApp {
    background: linear-gradient(135deg, var(--navy-darkest) 0%, var(--navy-dark) 40%, var(--navy-mid) 100%) !important;
    font-family: 'Exo 2', sans-serif;
    color: var(--text-primary) !important;
}
.stApp::before {
    content: '';
    position: fixed; inset: 0;
    background-image:
        radial-gradient(1px 1px at 20% 30%, rgba(0,212,255,.4) 0%, transparent 100%),
        radial-gradient(1px 1px at 80% 10%, rgba(255,255,255,.3) 0%, transparent 100%),
        radial-gradient(1px 1px at 50% 70%, rgba(0,212,255,.3) 0%, transparent 100%),
        radial-gradient(1px 1px at 10% 80%, rgba(125,211,252,.2) 0%, transparent 100%),
        radial-gradient(1px 1px at 90% 60%, rgba(255,255,255,.2) 0%, transparent 100%);
    pointer-events: none; z-index: 0;
}

[data-testid="column"],
[data-testid="column"] > div,
[data-testid="column"] > div > div,
[data-testid="stHorizontalBlock"],
[data-testid="stHorizontalBlock"] > div,
[data-testid="stVerticalBlock"],
[data-testid="stVerticalBlock"] > div,
div[class*="block-container"],
div[class*="element-container"],
.stHorizontalBlock,
.stHorizontalBlock > div,
.row-widget {
    background:       transparent !important;
    background-color: transparent !important;
    border:     none !important;
    box-shadow: none !important;
}

[data-testid="column"]:hover,
[data-testid="column"] > div:hover,
[data-testid="stHorizontalBlock"]:hover,
[data-testid="stHorizontalBlock"] > div:hover,
div[class*="element-container"]:hover {
    background:       transparent !important;
    background-color: transparent !important;
    border:     none !important;
    box-shadow: none !important;
}

[data-testid="column"]::before,            [data-testid="column"]::after,
[data-testid="stHorizontalBlock"]::before, [data-testid="stHorizontalBlock"]::after,
.stHorizontalBlock::before,               .stHorizontalBlock::after,
div[class*="element-container"]::before,  div[class*="element-container"]::after {
    display:    none !important;
    background: none !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--navy-dark) 0%, var(--navy-mid) 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

[data-testid="stSidebar"] .stButton > button {
    background:   rgba(13,34,71,0.70) !important;
    color:        var(--text-primary) !important;
    -webkit-text-fill-color: var(--text-primary) !important;
    border:       1px solid rgba(0,212,255,0.20) !important;
    border-radius: 6px !important;
    font-family:  'Exo 2', sans-serif !important;
    font-weight:  400 !important;
    text-align:   left !important;
    transition:   background .2s ease, border-color .2s ease !important;
    white-space:  nowrap !important;
    overflow:     hidden !important;
    text-overflow: ellipsis !important;
    -webkit-tap-highlight-color: transparent !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background:   rgba(30,77,183,0.50) !important;
    border-color: var(--cyan-glow) !important;
    color:        var(--cyan-soft) !important;
    -webkit-text-fill-color: var(--cyan-soft) !important;
    box-shadow:   0 0 10px rgba(0,212,255,0.20) !important;
    transform:    none !important;
}
[data-testid="stSidebar"] .stButton > button:focus,
[data-testid="stSidebar"] .stButton > button:active {
    background:   rgba(30,77,183,0.60) !important;
    border-color: var(--cyan-glow) !important;
    outline:      none !important;
    box-shadow:   0 0 0 2px rgba(0,212,255,0.30) !important;
}

[data-testid="stSidebar"] div[data-testid="stButton"]:first-of-type > button {
    background: linear-gradient(135deg, var(--navy-accent), var(--cyan-glow)) !important;
    color:      var(--navy-darkest) !important;
    -webkit-text-fill-color: var(--navy-darkest) !important;
    border:     none !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px;
}
[data-testid="stSidebar"] div[data-testid="stButton"]:first-of-type > button:hover {
    box-shadow: 0 0 18px var(--cyan-glow) !important;
}

[data-testid="stSidebar"] .stButton > button[title="Delete chat"] {
    background:  transparent !important;
    border:      1px solid rgba(255,80,80,0.25) !important;
    color:       #ff6b6b !important;
    -webkit-text-fill-color: #ff6b6b !important;
    padding:     2px 6px !important;
    font-size:   0.8rem !important;
}
[data-testid="stSidebar"] .stButton > button[title="Delete chat"]:hover {
    background:  rgba(255,80,80,0.15) !important;
    border-color: #ff6b6b !important;
    box-shadow:  none !important;
}

.history-date-label {
    font-size: 0.69rem; color: var(--text-muted);
    padding: 6px 4px 2px; text-transform: uppercase;
    letter-spacing: 0.08em; font-family: 'Exo 2', sans-serif;
}

/* ── Tabs styling ── */
[data-testid="stSidebar"] .stTabs [data-baseweb="tab-list"] {
    background: rgba(13,34,71,0.60) !important;
    border-radius: 10px !important;
    padding: 3px !important;
    gap: 2px !important;
    border: 1px solid rgba(0,212,255,0.18) !important;
}
[data-testid="stSidebar"] .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: 7px !important;
    font-family: 'Exo 2', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    padding: 6px 10px !important;
    border: none !important;
    flex: 1 !important;
    justify-content: center !important;
    transition: all .2s ease !important;
}
[data-testid="stSidebar"] .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--navy-accent), var(--navy-light)) !important;
    color: var(--cyan-glow) !important;
    -webkit-text-fill-color: var(--cyan-glow) !important;
    box-shadow: 0 0 10px rgba(0,212,255,0.3) !important;
}
[data-testid="stSidebar"] .stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}
[data-testid="stSidebar"] .stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

/* Video/Audio card */
.video-summary-card {
    background: rgba(13,34,71,0.75);
    border: 1px solid rgba(0,212,255,0.25);
    border-radius: 10px;
    padding: 12px 14px;
    margin-top: 8px;
}
.video-badge {
    display: inline-block;
    background: linear-gradient(135deg, #1e4db7, #00d4ff);
    color: #020b18 !important;
    -webkit-text-fill-color: #020b18 !important;
    font-size: 0.70rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 20px;
    margin-bottom: 6px;
    font-family: 'Exo 2', sans-serif;
}

div[data-testid="stToggle"],
div[data-testid="stToggle"] > div,
div[data-testid="stToggle"] > label,
div[class*="stToggle"],
div[class*="stToggle"] > div {
    background:       transparent !important;
    background-color: transparent !important;
    border:     none !important;
    box-shadow: none !important;
}

div[data-testid="stToggle"] input[type="checkbox"] + div,
div[data-testid="stToggle"] > label > div:first-child {
    background:   var(--navy-mid) !important;
    border:       1.5px solid rgba(0,212,255,0.35) !important;
    border-radius: 999px !important;
    transition:   background .25s ease, border-color .25s ease !important;
}
div[data-testid="stToggle"] input[type="checkbox"]:checked + div,
div[data-testid="stToggle"] > label > div.checked {
    background:   var(--navy-accent) !important;
    border-color: var(--cyan-glow) !important;
    box-shadow:   0 0 8px rgba(0,212,255,0.45) !important;
}
div[data-testid="stToggle"] input[type="checkbox"] + div > div {
    background: var(--cyan-soft) !important;
}
div[data-testid="stToggle"] > label > span,
div[data-testid="stToggle"] p,
div[data-testid="stToggle"] label {
    color: var(--text-secondary) !important;
    -webkit-text-fill-color: var(--text-secondary) !important;
    font-family: 'Exo 2', sans-serif !important;
    font-size: 0.82rem !important;
}

h1 {
    font-family: 'Exo 2', sans-serif !important; font-weight: 700 !important;
    background: linear-gradient(90deg, var(--cyan-glow), var(--cyan-soft), var(--gold));
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    text-align: center !important;
    width: 100% !important;
}
h2,h3,h4,h5 { color: var(--cyan-soft) !important; font-family:'Exo 2',sans-serif !important; }

[data-testid="stChatMessage"] {
    background:    var(--card-bg) !important;
    border:        1px solid var(--border) !important;
    border-radius: 12px !important;
    box-shadow:    var(--shadow) !important;
    backdrop-filter: blur(12px) !important;
    margin-bottom: 10px !important;
    padding:       4px 8px !important;
}

[data-testid="stChatInput"] textarea {
    color: #0d2247 !important; -webkit-text-fill-color: #0d2247 !important;
    caret-color: #1e4db7 !important; background: #e8f4fd !important; border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #4d7a9e !important; -webkit-text-fill-color: #4d7a9e !important;
}
[data-testid="stChatInput"] {
    background: #e8f4fd !important;
    border: 1.5px solid var(--cyan-glow) !important; border-radius: 14px !important;
}

.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
input[type="text"], input[type="url"], input[type="search"] {
    color: #e8f4fd !important; -webkit-text-fill-color: #e8f4fd !important;
    caret-color: var(--cyan-glow) !important;
    background: var(--navy-mid) !important;
    border: 1px solid var(--border) !important; border-radius: 8px !important;
    font-family: 'Exo 2', sans-serif !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--cyan-glow) !important; box-shadow: 0 0 8px rgba(0,212,255,.3) !important;
}
.stTextInput > div > div > input::placeholder,
.stTextArea > div > div > textarea::placeholder {
    color: var(--text-muted) !important; -webkit-text-fill-color: var(--text-muted) !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: var(--navy-mid) !important;
    border: 2px dashed rgba(0,212,255,.35) !important; border-radius: 10px !important;
    color: var(--text-secondary) !important;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--cyan-glow) !important; }
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, var(--navy-accent), var(--navy-light)) !important;
    color: var(--cyan-soft) !important; -webkit-text-fill-color: var(--cyan-soft) !important;
    border: 1px solid rgba(0,212,255,.4) !important; border-radius: 8px !important;
    font-family: 'Exo 2', sans-serif !important; font-weight: 600 !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background: linear-gradient(135deg, var(--navy-light), var(--navy-accent)) !important;
    border-color: var(--cyan-glow) !important; box-shadow: 0 0 12px rgba(0,212,255,.35) !important;
}
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small {
    color: var(--text-secondary) !important; -webkit-text-fill-color: var(--text-secondary) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--navy-accent) 0%, var(--navy-light) 100%) !important;
    color: var(--text-primary) !important; -webkit-text-fill-color: var(--text-primary) !important;
    border: 1px solid var(--border) !important; border-radius: 8px !important;
    font-family: 'Exo 2', sans-serif !important; font-weight: 600 !important;
    transition: all .25s ease !important;
}
.stButton > button:hover {
    border-color: var(--cyan-glow) !important;
    background: linear-gradient(135deg, var(--navy-light) 0%, var(--navy-accent) 100%) !important;
    box-shadow: 0 0 14px rgba(0,212,255,.35) !important; transform: translateY(-1px) !important;
    color: var(--cyan-soft) !important; -webkit-text-fill-color: var(--cyan-soft) !important;
}

div[data-testid="column"] .stButton > button {
    background: rgba(13,34,71,0.80) !important;
    color: var(--cyan-soft) !important; -webkit-text-fill-color: var(--cyan-soft) !important;
    border: 1px solid rgba(0,212,255,.30) !important;
    font-size: 0.82rem !important; font-weight: 400 !important;
    white-space: normal !important; text-align: center !important; min-height: 44px !important;
}
div[data-testid="column"] .stButton > button:hover {
    background: rgba(30,77,183,0.60) !important; border-color: var(--cyan-glow) !important;
    color: var(--text-primary) !important; -webkit-text-fill-color: var(--text-primary) !important;
    box-shadow: 0 0 12px rgba(0,212,255,.30) !important; transform: translateY(-1px) !important;
}

[data-testid="stDownloadButton"],
[data-testid="stDownloadButton"] > div,
[data-testid="stDownloadButton"] > div > div {
    display: block !important; visibility: visible !important;
    opacity: 1 !important; width: 100% !important; pointer-events: auto !important;
}
[data-testid="stDownloadButton"] button {
    display: flex !important; align-items: center !important; justify-content: center !important;
    visibility: visible !important; opacity: 1 !important; pointer-events: auto !important;
    background: linear-gradient(135deg, #0f3460 0%, var(--navy-accent) 100%) !important;
    color: var(--cyan-soft) !important; -webkit-text-fill-color: var(--cyan-soft) !important;
    border: 1.5px solid var(--cyan-glow) !important; border-radius: 8px !important;
    font-family: 'Exo 2', sans-serif !important; font-weight: 600 !important; font-size: .88rem !important;
    width: 100% !important; min-height: 40px !important; padding: 8px 16px !important;
    cursor: pointer !important; transition: all .25s ease !important;
    z-index: 10 !important; position: relative !important;
}
[data-testid="stDownloadButton"] button:hover {
    box-shadow: 0 0 18px rgba(0,212,255,.45) !important; transform: translateY(-1px) !important;
    background: linear-gradient(135deg, var(--navy-accent), #0f3460) !important;
}

.download-card {
    background: rgba(13,34,71,0.70);
    border: 1px solid rgba(0,212,255,0.22);
    border-radius: 10px; padding: 14px 16px 10px; margin-top: 14px;
}
.download-card h5 {
    color: var(--cyan-soft) !important; font-size: 0.88rem !important;
    margin-bottom: 10px !important; font-family: 'Exo 2', sans-serif !important;
}

[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, var(--cyan-glow), var(--gold)) !important; border-radius: 4px !important;
}
.stRadio > div { background: transparent !important; }
.stRadio label, .stCheckbox label, .stToggle label { color: var(--text-primary) !important; }
[data-testid="stAlert"] { border-radius: 8px !important; backdrop-filter: blur(8px) !important; }
.stSuccess { background: rgba(0,180,100,.12) !important; border: 1px solid rgba(0,220,120,.35) !important; color: #6effc5 !important; }
.stError   { background: rgba(220,50,50,.12)  !important; border: 1px solid rgba(255,80,80,.35)   !important; }
.stInfo    { background: rgba(0,100,200,.15)  !important; border: 1px solid var(--border) !important; color: var(--cyan-soft) !important; }
.stWarning { background: rgba(200,150,0,.12)  !important; border: 1px solid rgba(251,191,36,.35)  !important; color: var(--gold) !important; }
.stCaption, small, .caption { color: var(--text-muted) !important; }
hr { border-color: var(--border) !important; }
[data-testid="stSpinner"] { color: var(--cyan-glow) !important; }
[data-testid="stSelectbox"] > div > div {
    background: var(--navy-mid) !important; border: 1px solid var(--border) !important;
    color: var(--text-primary) !important; border-radius: 8px !important;
}
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--navy-darkest); }
::-webkit-scrollbar-thumb { background: var(--navy-accent); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--cyan-glow); }
p, li, span, label, div { color: var(--text-primary); }
.stMarkdown p { color: var(--text-primary) !important; }
.stMarkdown code {
    background: rgba(0,212,255,.10) !important; color: var(--cyan-glow) !important;
    font-family: 'JetBrains Mono', monospace !important; border-radius: 4px !important; padding: 2px 6px !important;
}
.stMarkdown pre {
    background: var(--navy-darkest) !important; border: 1px solid var(--border) !important; border-radius: 8px !important;
}
button:focus, button:active { background-clip: padding-box !important; outline: none !important; }
[data-testid="baseButton-secondary"]:hover::after,
[data-testid="baseButton-secondary"]:focus::after { display: none !important; }

body div[role="tooltip"] {
    background: rgba(13, 34, 71, 0.98) !important;
    color: #7dd3fc !important;
    border: 1px solid rgba(0,212,255,0.5) !important;
    border-radius: 8px !important;
    font-family: 'Exo 2', sans-serif !important;
    font-size: 0.75rem !important;
    padding: 6px 10px !important;
    box-shadow: 0 0 12px rgba(0,212,255,0.35) !important;
    backdrop-filter: blur(8px) !important;
    z-index: 999999 !important;
}
body div[role="tooltip"]::after {
    border-top-color: rgba(13, 34, 71, 0.98) !important;
}
body div[role="tooltip"] * {
    background: transparent !important;
    color: #7dd3fc !important;
}

[data-testid="stTooltipHoverTarget"]::after,
[data-testid="stTooltipHoverTarget"]::before {
    display: none !important;
}
[title] { pointer-events: auto; }

div[data-testid="stButton"] button {
    background: rgba(13, 34, 71, 0.85) !important;
    color: #7dd3fc !important;
    border: 1px solid rgba(0,212,255,0.3) !important;
    border-radius: 10px !important;
    text-align: left !important;
    padding: 10px 14px !important;
    font-size: 14px !important;
}
div[data-testid="stButton"] button:hover {
    background: rgba(0, 212, 255, 0.15) !important;
    color: #ffffff !important;
    border: 1px solid rgba(0,212,255,0.6) !important;
}
div[data-testid="stButton"] button:active {
    background: rgba(0, 212, 255, 0.25) !important;
}
div[data-testid="stButton"] button div {
    color: inherit !important;
    background: transparent !important;
}
</style>
"""
st.markdown(NAVY_CSS, unsafe_allow_html=True)
st.title("🧩ThinkHub")


# ══════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════
session_defaults = {
    "chroma_path": "./chroma_db",
    "model_version": "deepseek",
    "vector_store": None,
    "processed_documents": [],
    "history": [],
    "conversations": {},
    "current_conv_id": None,
    "use_web_search": False,
    "force_web_search": False,
    "similarity_threshold": 0.7,
    "rag_enabled": True,
    "quiz_data": None,
    "quiz_answers": {},
    "quiz_submitted": False,
    "last_response": "",
    "last_question": "",
    "suggestion_clicked": None,
    "cached_suggestions": [],
    "suggestions_for_msg": "",
    "dl_pdf_bytes":        None,
    "dl_audio_bytes":      None,
    "dl_pdf_error":        None,
    "dl_audio_error":      None,
    "dl_timestamp":        "",
    "video_summary":       "",
    "video_file_name":     "",
    "video_gemini_name":   "",
    "video_active":        False,
}
for k, v in session_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.current_conv_id is None:
    st.session_state.current_conv_id = str(uuid.uuid4())


# ══════════════════════════════════════════════
# CONVERSATION HELPERS
# ══════════════════════════════════════════════
def _save_current_conversation():
    cid = st.session_state.current_conv_id
    if not cid or not st.session_state.history:
        return
    first_user = next(
        (m["content"] for m in st.session_state.history if m["role"] == "user"),
        "Untitled Chat",
    )
    title = first_user[:48] + ("…" if len(first_user) > 48 else "")
    existing = st.session_state.conversations.get(cid, {})
    st.session_state.conversations[cid] = {
        "title":      title,
        "messages":   list(st.session_state.history),
        "created_at": existing.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M")),
    }
    if len(st.session_state.conversations) > 10:
        sorted_convs = sorted(
            st.session_state.conversations.items(),
            key=lambda x: x[1].get("created_at", "")
        )
        while len(sorted_convs) > 10:
            oldest_id, _ = sorted_convs.pop(0)
            del st.session_state.conversations[oldest_id]


def _reset_chat_state():
    st.session_state.quiz_data           = None
    st.session_state.quiz_answers        = {}
    st.session_state.quiz_submitted      = False
    st.session_state.suggestion_clicked  = None
    st.session_state.cached_suggestions  = []
    st.session_state.suggestions_for_msg = ""
    st.session_state.dl_pdf_bytes        = None
    st.session_state.dl_audio_bytes      = None
    st.session_state.dl_pdf_error        = None
    st.session_state.dl_audio_error      = None
    st.session_state.last_response       = ""
    st.session_state.last_question       = ""


def new_conversation():
    _save_current_conversation()
    st.session_state.current_conv_id = str(uuid.uuid4())
    st.session_state.history = []
    _reset_chat_state()


def load_conversation(conv_id: str):
    _save_current_conversation()
    conv = st.session_state.conversations.get(conv_id)
    if conv:
        st.session_state.current_conv_id = conv_id
        st.session_state.history = list(conv["messages"])
        _reset_chat_state()


def delete_conversation(conv_id: str):
    if conv_id in st.session_state.conversations:
        del st.session_state.conversations[conv_id]
    if st.session_state.current_conv_id == conv_id:
        new_conversation()


def filter_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


# ══════════════════════════════════════════════
# CHROMADB
# ══════════════════════════════════════════════
def init_chroma():
    chroma = ChromaDb(
        collection=COLLECTION_NAME,
        path=st.session_state.chroma_path,
        embedder=EMBEDDING_MODEL,
        persistent_client=True,
    )
    try:
        chroma.client.get_collection(name=COLLECTION_NAME)
    except Exception:
        chroma.create()
    return chroma


# ══════════════════════════════════════════════
# DOCUMENT PROCESSING
# ══════════════════════════════════════════════
def split_texts(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return [
        Document(page_content=c.page_content, metadata=c.metadata)
        for c in splitter.split_documents(documents)
        if c.page_content.strip()
    ]


def process_pdf(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            docs = PyPDFLoader(tmp.name).load()
        for d in docs:
            d.metadata.update({"source_type": "pdf", "file_name": uploaded_file.name,
                                "page": d.metadata.get("page", "unknown")})
        return split_texts(docs)
    except Exception as e:
        st.error(f"PDF processing error: {e}")
        return []


def process_web(url: str):
    try:
        docs = WebBaseLoader(url).load()
        if not docs:
            st.sidebar.error("No content extracted from the URL.")
            return []
        for d in docs:
            d.metadata.update({"source": url, "timestamp": datetime.now().isoformat()})
        return split_texts(docs)
    except Exception as e:
        st.error(f"Web processing error: {e}")
        return []


# ══════════════════════════════════════════════
# VIDEO / AUDIO PROCESSING
# ══════════════════════════════════════════════
SUPPORTED_VIDEO_TYPES = {
    ".mp4":  "video/mp4",
    ".mov":  "video/quicktime",
    ".avi":  "video/x-msvideo",
    ".mkv":  "video/x-matroska",
    ".webm": "video/webm",
    ".flv":  "video/x-flv",
    ".wmv":  "video/x-ms-wmv",
    ".mp3":  "audio/mpeg",
}


def _upload_to_gemini(tmp_path: str, mime_type: str):
    """Upload file to Gemini Files API and wait until ACTIVE."""
    uploaded = genai.upload_file(path=tmp_path, mime_type=mime_type)
    max_wait, waited = 300, 0
    while uploaded.state.name == "PROCESSING" and waited < max_wait:
        time.sleep(4)
        uploaded = genai.get_file(uploaded.name)
        waited += 4
    if uploaded.state.name != "ACTIVE":
        raise RuntimeError(f"File processing timed out (state: {uploaded.state.name})")
    return uploaded


def process_video(uploaded_file) -> tuple[str, str]:
    """Upload video/audio to Gemini and return (gemini_file_name, summary_text)."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower() or ".mp4"
    mime   = SUPPORTED_VIDEO_TYPES.get(suffix, "video/mp4")
    is_audio = suffix == ".mp3"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        gem_file = _upload_to_gemini(tmp_path, mime)
        model    = genai.GenerativeModel(model_name=FAST_MODEL)

        if is_audio:
            prompt = (
                "Provide a thorough, structured summary of this audio file. Include:\n"
                "1. **Overview** — what the audio is about in 2–3 sentences.\n"
                "2. **Key Topics & Points** — bullet list of every major topic or discussion point.\n"
                "3. **Important Details** — any names, facts, statistics, or examples mentioned.\n"
                "4. **Conclusions / Takeaways** — main messages or outcomes.\n\n"
                "Be thorough — your summary will be used to answer detailed questions about the audio."
            )
        else:
            prompt = (
                "Provide a thorough, structured summary of this video. Include:\n"
                "1. **Overview** — what the video is about in 2–3 sentences.\n"
                "2. **Key Topics & Points** — bullet list of every major topic, concept, "
                "   fact, or demonstration covered.\n"
                "3. **Important Details** — any names, statistics, examples, or steps mentioned.\n"
                "4. **Conclusions / Takeaways** — main messages or outcomes.\n\n"
                "Be thorough — your summary will be used to answer detailed questions about the video."
            )

        summary = model.generate_content([gem_file, prompt])
        return gem_file.name, summary.text
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def answer_video_question(question: str, history_context: str) -> str:
    """Answer a question about the uploaded video/audio."""
    gemini_name = st.session_state.video_gemini_name
    summary     = st.session_state.video_summary

    qa_prompt = (
        f"{history_context}\n\n" if history_context else ""
    ) + f"Question about the file: {question}\n\nAnswer in detail."

    # Try live Gemini file Q&A first
    if gemini_name:
        try:
            gem_file = genai.get_file(gemini_name)
            if gem_file.state.name == "ACTIVE":
                model = genai.GenerativeModel(model_name=FAST_MODEL)
                resp  = model.generate_content([gem_file, qa_prompt])
                return resp.text
        except Exception:
            pass

    # Fallback to summary-based Q&A
    if summary:
        return Agent(
            name="Media QA Agent", model=Gemini(id=FAST_MODEL),
            instructions=(
                "You are a media analysis assistant. Answer questions using the "
                "provided summary. Be specific and detailed."
            ),
            markdown=True,
        ).run(f"File Summary:\n{summary}\n\n{qa_prompt}").content

    return "⚠️ No media context available. Please upload a video or audio file first."


# ══════════════════════════════════════════════
# RETRIEVAL
# ══════════════════════════════════════════════
def retrieve_documents(prompt, vector_store, collection_name, similarity_threshold):
    collection = vector_store.client.get_collection(name=collection_name)
    results = collection.query(
        query_texts=[prompt], n_results=5,
        include=["documents", "metadatas", "distances"],
    )
    docs, sources = [], []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        if dist < similarity_threshold:
            docs.append(doc)
            sources.append(meta)
    return docs, sources, len(docs) > 0


# ══════════════════════════════════════════════
# AGENTS
# ══════════════════════════════════════════════
def get_web_search_agent():
    return Agent(
        name="Web Search Agent", model=Gemini(id=FAST_MODEL),
        tools=[DuckDuckGoTools()],
        instructions="Search DuckDuckGo for reliable info. Summarise key points clearly.",
        markdown=True,
    )


def get_rag_agent():
    return Agent(
        name="RAG Agent", model=Gemini(id=FAST_MODEL),
        instructions=(
            "You are an intelligent AI assistant. Use the provided context and conversation "
            "history to answer accurately. If the answer is in the context, use it. "
            "For follow-up questions, refer back to the conversation history for continuity. "
            "Never mention context, documents, or retrieval. Be clear and accurate."
        ),
        markdown=True,
    )


def get_suggestion_agent():
    return Agent(
        name="Suggestion Agent", model=Gemini(id=FAST_MODEL),
        instructions=(
            "Generate exactly 4 short follow-up question suggestions DIRECTLY related to the "
            "topics already discussed in the conversation. The questions must be answerable "
            "from the same subject matter as the conversation. "
            'Return ONLY a JSON array of 4 strings. No preamble, no markdown. '
            'Example: ["What is X?","How does Y work?","Compare A and B","Why is Z important?"] '
            "Keep each under 10 words."
        ),
    )


# ══════════════════════════════════════════════
# SUGGESTIONS (cached)
# ══════════════════════════════════════════════
def get_or_generate_suggestions() -> list:
    last_msg = st.session_state.history[-1]["content"] if st.session_state.history else ""
    if st.session_state.cached_suggestions and st.session_state.suggestions_for_msg == last_msg:
        return st.session_state.cached_suggestions
    try:
        ctx = "\n".join(
            f"{'User' if m['role']=='user' else 'Assistant'}: {m['content'][:400]}"
            for m in st.session_state.history[-8:]
        )
        result = get_suggestion_agent().run(
            f"Conversation so far:\n{ctx}\n\n"
            "Generate 4 follow-up questions that are directly relevant to the topics above."
        ).content.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\[.*?\]", result, re.DOTALL)
        suggestions = json.loads(match.group())[:4] if match else []
        if not suggestions:
            raise ValueError("empty")
    except Exception:
        suggestions = ["Explain this in detail", "Give examples",
                       "What are the benefits?", "Compare with alternatives"]
    st.session_state.cached_suggestions  = suggestions
    st.session_state.suggestions_for_msg = last_msg
    return suggestions


# ══════════════════════════════════════════════
# QUIZ HELPERS
# ══════════════════════════════════════════════
def generate_quiz(text: str) -> str:
    return Agent(
        name="Quiz Generator", model=Gemini(id=FAST_MODEL),
        instructions=(
            "Generate 5 MCQs. Return ONLY a JSON array, no markdown. "
            'Format: [{"question":"...","options":["A.","B.","C.","D."],"answer":"A.","explanation":"..."}]'
        ),
    ).run(text).content


def extract_json(text: str):
    text = text.replace("```json", "").replace("```", "").strip()
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        return json.loads(m.group())
    raise ValueError("No JSON array found.")


def build_quiz_context() -> str:
    return "\n\n".join(
        f"{'Question' if m['role']=='user' else 'Answer'}: {m['content']}"
        for m in st.session_state.history
    )


# ══════════════════════════════════════════════
# PDF / AUDIO GENERATORS — FULL CONVERSATION
# ══════════════════════════════════════════════
def _safe_para(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def generate_full_conversation_pdf(history: list) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        rightMargin=60, leftMargin=60, topMargin=60, bottomMargin=60,
    )
    styles = getSampleStyleSheet()
    style_title = ParagraphStyle(
        "Title2", parent=styles["Title"],
        fontSize=18, textColor=colors.HexColor("#1a73e8"), spaceAfter=6,
    )
    style_meta = ParagraphStyle(
        "Meta", parent=styles["Normal"],
        fontSize=9, textColor=colors.grey, spaceAfter=18,
    )
    style_q_label = ParagraphStyle(
        "QLabel", parent=styles["Normal"],
        fontSize=10, textColor=colors.HexColor("#1a73e8"),
        fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=4,
    )
    style_q_text = ParagraphStyle(
        "QText", parent=styles["Normal"],
        fontSize=11, textColor=colors.HexColor("#1a1a1a"),
        leading=16, spaceAfter=6, leftIndent=10,
    )
    style_a_label = ParagraphStyle(
        "ALabel", parent=styles["Normal"],
        fontSize=10, textColor=colors.HexColor("#0b6b3a"),
        fontName="Helvetica-Bold", spaceBefore=4, spaceAfter=4,
    )
    style_a_text = ParagraphStyle(
        "AText", parent=styles["Normal"],
        fontSize=11, textColor=colors.HexColor("#1a1a1a"),
        leading=18, spaceAfter=6, leftIndent=10,
    )

    story: list = [
        Paragraph("ThinkHub — Full Conversation", style_title),
        Paragraph(
            f"Exported on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
            f"{len([m for m in history if m['role']=='user'])} question(s)",
            style_meta,
        ),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a73e8"), spaceAfter=10),
    ]

    q_num = 0
    for msg in history:
        role    = msg.get("role", "")
        content = filter_think_tags(msg.get("content", "")).strip()
        if not content:
            continue
        if role == "user":
            q_num += 1
            story.append(Paragraph(f"Q{q_num}  USER", style_q_label))
            for line in content.split("\n"):
                line = line.strip()
                if line:
                    story.append(Paragraph(_safe_para(line), style_q_text))
        elif role == "assistant":
            story.append(Paragraph("ASSISTANT", style_a_label))
            for line in content.split("\n"):
                line = line.strip()
                if line:
                    story.append(Paragraph(_safe_para(line), style_a_text))
            story.append(
                HRFlowable(width="100%", thickness=0.5,
                           color=colors.HexColor("#cccccc"), spaceAfter=6)
            )

    doc.build(story)
    return buf.getvalue()


def generate_full_conversation_audio(history: list) -> bytes:
    from gtts import gTTS
    lines = []
    q_num = 0
    for msg in history:
        role    = msg.get("role", "")
        content = filter_think_tags(msg.get("content", "")).strip()
        if not content:
            continue
        if role == "user":
            q_num += 1
            lines.append(f"Question {q_num}. {content}")
        elif role == "assistant":
            lines.append(f"Answer. {content}")
    full_text = "  ...  ".join(lines)
    buf = io.BytesIO()
    gTTS(text=full_text, lang="en").write_to_fp(buf)
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
st.sidebar.markdown("## 💬 Chat History")

if st.sidebar.button("✏️  New Chat", key="new_chat_btn", use_container_width=True):
    new_conversation()
    st.rerun()

st.sidebar.markdown(
    "<hr style='border-color:rgba(0,212,255,.15);margin:6px 0 8px'>",
    unsafe_allow_html=True,
)

_save_current_conversation()

if st.session_state.conversations:
    today_str = datetime.now().strftime("%Y-%m-%d")
    try:
        yesterday_str = datetime.now().replace(day=datetime.now().day - 1).strftime("%Y-%m-%d")
    except Exception:
        yesterday_str = None

    groups: dict = {"Today": [], "Yesterday": [], "Older": []}
    for cid, conv in sorted(
        st.session_state.conversations.items(),
        key=lambda x: x[1].get("created_at", ""),
        reverse=True,
    ):
        d = conv.get("created_at", "")[:10]
        if d == today_str:
            groups["Today"].append((cid, conv))
        elif yesterday_str and d == yesterday_str:
            groups["Yesterday"].append((cid, conv))
        else:
            groups["Older"].append((cid, conv))

    for label, items in groups.items():
        if not items:
            continue
        st.sidebar.markdown(
            f"<div class='history-date-label'>{label}</div>",
            unsafe_allow_html=True,
        )
        for cid, conv in items:
            is_active = cid == st.session_state.current_conv_id
            q_count   = len([m for m in conv["messages"] if m["role"] == "user"])
            icon      = "🔵 " if is_active else "💬 "
            col_load, col_del = st.sidebar.columns([0.82, 0.18])
            with col_load:
                if st.button(
                    f"{icon}{conv['title']}",
                    key=f"hist_{cid}",
                    use_container_width=True,
                    help=f"{q_count} message(s) · {conv.get('created_at','')}",
                ):
                    load_conversation(cid)
                    st.rerun()
            with col_del:
                if st.button("🗑", key=f"del_{cid}", help="Delete chat"):
                    delete_conversation(cid)
                    st.rerun()
else:
    st.sidebar.caption("No previous chats yet. Start asking questions!")

st.sidebar.markdown(
    "<hr style='border-color:rgba(0,212,255,.15);margin:10px 0'>",
    unsafe_allow_html=True,
)

# ── Agent Config ──────────────────────────────────────────────────
st.sidebar.header("🤖 AI Settings")
st.session_state.model_version = st.sidebar.radio(
    "Choose an AI Model", ["gemini-2.5-flash"], help="Gemini Model is used."
)

st.sidebar.header("🔍 Document Intelligence")
st.session_state.rag_enabled = st.sidebar.toggle(
    "Use My Documents", value=st.session_state.rag_enabled
)

if st.sidebar.button("🗑️ Clear Current Chat"):
    cid = st.session_state.current_conv_id
    st.session_state.history = []
    _reset_chat_state()
    if cid in st.session_state.conversations:
        del st.session_state.conversations[cid]
    st.rerun()

st.sidebar.header("🌐 Web Search Access")
st.session_state.use_web_search = st.sidebar.checkbox(
    "Enable Internet if Needed", value=st.session_state.use_web_search
)

# ── Quiz Generator ────────────────────────────────────────────────
st.sidebar.header("📝 Quiz Generator")
if st.sidebar.button("Generate Quiz"):
    if st.session_state.history:
        with st.spinner("Generating quiz..."):
            quiz_text = generate_quiz(build_quiz_context())
        try:
            st.session_state.quiz_data      = extract_json(quiz_text)
            st.session_state.quiz_answers   = {}
            st.session_state.quiz_submitted = False
            st.sidebar.success("Quiz ready! Scroll down.")
        except Exception as e:
            st.sidebar.error(f"Quiz failed: {e}")
            st.sidebar.write(quiz_text)
    else:
        st.sidebar.warning("Ask some questions first.")


# ══════════════════════════════════════════════════════════════════
# ── HORIZONTAL TABS: 📄 PDF | 🌐 URL | 🎬 Video ──────────────────
# ══════════════════════════════════════════════════════════════════
if st.session_state.rag_enabled:
    chroma_client = init_chroma()
    st.sidebar.header("📁 Upload Your Files")

    tab_pdf, tab_url, tab_video = st.sidebar.tabs(["📄 PDF", "🌐 URL", "🎬 Video"])

    # ── Tab 1: PDF ────────────────────────────────────────────────
    with tab_pdf:
        uploaded_file = st.file_uploader(
            "Upload PDF", type=["pdf"], key="pdf_uploader",
            label_visibility="collapsed",
        )
        if uploaded_file and uploaded_file.name not in st.session_state.processed_documents:
            data = process_pdf(uploaded_file)
            if data:
                ids = [str(uuid.uuid4()) for _ in data]
                col = chroma_client.client.get_collection(name=COLLECTION_NAME)
                col.add(
                    ids=ids,
                    documents=[d.page_content for d in data],
                    metadatas=[d.metadata for d in data],
                )
                st.session_state.processed_documents.append(uploaded_file.name)
                st.success(f"✅ {uploaded_file.name} processed!")

    # ── Tab 2: URL ────────────────────────────────────────────────
    with tab_url:
        web_url = st.text_input(
            "Enter a URL", placeholder="https://example.com",
            key="url_input", label_visibility="collapsed",
        )
        if web_url and st.button("Process URL", key="process_url_btn"):
            if web_url not in st.session_state.processed_documents:
                texts = process_web(web_url)
                if texts:
                    ids = [str(uuid.uuid4()) for _ in texts]
                    col = chroma_client.client.get_collection(name=COLLECTION_NAME)
                    col.add(
                        ids=ids,
                        documents=[d.page_content for d in texts],
                        metadatas=[d.metadata for d in texts],
                    )
                    st.session_state.processed_documents.append(web_url)
                    st.success("✅ URL processed!")
            else:
                st.info("This URL has already been processed.")

    # ── Tab 3: Video / Audio ──────────────────────────────────────
    with tab_video:

        # Supported formats card — always visible
        st.markdown(
            """
            <div style='
                background: rgba(0,212,255,0.06);
                border: 1px solid rgba(0,212,255,0.22);
                border-radius: 8px;
                padding: 9px 12px 8px;
                margin-bottom: 10px;
                font-family: "Exo 2", sans-serif;
                font-size: 0.74rem;
                line-height: 1.9;
            '>
                <span style='color:#00d4ff;font-weight:700;font-size:0.78rem;'>
                    📋 Supported Formats
                </span><br>
                <span style='color:#fbbf24;'>🎬 Video:</span>&nbsp;
                <code style='background:rgba(0,212,255,.12);color:#7dd3fc;padding:1px 5px;border-radius:4px;font-size:0.72rem'>.mp4</code>
                <code style='background:rgba(0,212,255,.12);color:#7dd3fc;padding:1px 5px;border-radius:4px;font-size:0.72rem'>.mov</code>
                <code style='background:rgba(0,212,255,.12);color:#7dd3fc;padding:1px 5px;border-radius:4px;font-size:0.72rem'>.avi</code>
                <code style='background:rgba(0,212,255,.12);color:#7dd3fc;padding:1px 5px;border-radius:4px;font-size:0.72rem'>.mkv</code>
                <code style='background:rgba(0,212,255,.12);color:#7dd3fc;padding:1px 5px;border-radius:4px;font-size:0.72rem'>.webm</code>
                <code style='background:rgba(0,212,255,.12);color:#7dd3fc;padding:1px 5px;border-radius:4px;font-size:0.72rem'>.flv</code>
                <code style='background:rgba(0,212,255,.12);color:#7dd3fc;padding:1px 5px;border-radius:4px;font-size:0.72rem'>.wmv</code>
                <br>
                <span style='color:#fbbf24;'>🎵 Audio:</span>&nbsp;
                <code style='background:rgba(0,212,255,.12);color:#7dd3fc;padding:1px 5px;border-radius:4px;font-size:0.72rem'>.mp3</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_video = st.file_uploader(
            "Upload Video / Audio",
            type=["mp4", "mov", "avi", "mkv", "webm", "flv", "wmv", "mp3"],
            key="video_uploader",
            label_visibility="collapsed",
        )

        if uploaded_video:
            if uploaded_video.name != st.session_state.video_file_name:
                is_audio    = uploaded_video.name.lower().endswith(".mp3")
                spinner_msg = (
                    "🎵 Uploading & analysing audio… this may take a moment."
                    if is_audio else
                    "🎬 Uploading & analysing video… this may take a minute."
                )
                with st.spinner(spinner_msg):
                    try:
                        gem_name, summary = process_video(uploaded_video)
                        st.session_state.video_gemini_name = gem_name
                        st.session_state.video_summary     = summary
                        st.session_state.video_file_name   = uploaded_video.name
                        st.session_state.video_active      = True

                        media_label = "Audio" if is_audio else "Video"
                        intro = (
                            f"{'🎵' if is_audio else '🎬'} **{media_label} uploaded:** "
                            f"`{uploaded_video.name}`\n\n"
                            f"Here's my summary:\n\n{summary}\n\n"
                            f"---\n*You can now ask me anything about this {media_label.lower()}!*"
                        )
                        st.session_state.history.append({"role": "assistant", "content": intro})
                        st.session_state.last_response = intro
                        _save_current_conversation()
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Processing failed: {e}")
            else:
                is_audio = uploaded_video.name.lower().endswith(".mp3")
                st.markdown(
                    f"<div class='video-summary-card'>"
                    f"<span class='video-badge'>{'🎵' if is_audio else '🎬'} LOADED</span><br>"
                    f"<small style='color:#94b8d4'>{uploaded_video.name}</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if st.button("🔄 Re-analyse", key="reanalyse_btn"):
                    st.session_state.video_file_name = ""
                    st.rerun()

        # Active file indicator
        if st.session_state.video_active and st.session_state.video_file_name:
            is_audio = st.session_state.video_file_name.lower().endswith(".mp3")
            st.markdown(
                f"<div class='video-summary-card'>"
                f"<span class='video-badge'>✅ ACTIVE</span><br>"
                f"<small style='color:#94b8d4'>{st.session_state.video_file_name}</small>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if st.button("🗑 Clear File", key="clear_video_btn"):
                st.session_state.video_summary     = ""
                st.session_state.video_file_name   = ""
                st.session_state.video_gemini_name = ""
                st.session_state.video_active      = False
                st.rerun()

else:
    chroma_client = init_chroma()


# ══════════════════════════════════════════════
# MAIN CHAT AREA
# ══════════════════════════════════════════════
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.write(filter_think_tags(message["content"]))

# ── Suggestions ───────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("**💡 Suggested questions:**")
    suggestions = get_or_generate_suggestions()
    cols = st.columns(len(suggestions))
    for idx, (col, sug) in enumerate(zip(cols, suggestions)):
        with col:
            if st.button(sug, key=f"sugg_{idx}_{hash(sug) & 0xFFFFFF}",
                         use_container_width=True):
                st.session_state.suggestion_clicked = sug
                st.rerun()

# ── Input row ─────────────────────────────────────────────────────
chat_col, toggle_col = st.columns([0.89, 0.11])
with chat_col:
    if st.session_state.video_active and st.session_state.video_file_name:
        placeholder = f"Ask about '{st.session_state.video_file_name}'…"
    elif st.session_state.rag_enabled:
        placeholder = "Ask your question…"
    else:
        placeholder = "Ask me anything…"
    text_prompt = st.chat_input(placeholder)
with toggle_col:
    st.markdown("<div style='padding-top:8px'>", unsafe_allow_html=True)
    st.session_state.force_web_search = st.toggle(
        "🌐", help="Always Use Internet", key="web_toggle"
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ── Resolve prompt ────────────────────────────────────────────────
prompt = None
if st.session_state.suggestion_clicked:
    prompt = st.session_state.suggestion_clicked
    st.session_state.suggestion_clicked = None
elif text_prompt:
    prompt = text_prompt


# ══════════════════════════════════════════════
# PROCESS QUERY
# ══════════════════════════════════════════════
if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.history.append({"role": "user", "content": prompt})
    st.session_state.last_question = prompt

    st.session_state.cached_suggestions  = []
    st.session_state.suggestions_for_msg = ""

    prior_turns     = st.session_state.history[:-1][-8:]
    history_context = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: "
        f"{filter_think_tags(m['content'])[:400]}"
        for m in prior_turns
    )

    response = ""

    # Priority 1: Video / Audio mode
    if st.session_state.video_active and st.session_state.video_summary:
        is_audio = st.session_state.video_file_name.lower().endswith(".mp3")
        with st.spinner("🎵 Analysing audio…" if is_audio else "🎬 Analysing video…"):
            try:
                response = answer_video_question(prompt, history_context)
            except Exception as ve:
                e = str(ve)
                if "429" in e or "quota" in e.lower():
                    response = "⚠️ **API Quota Exceeded** — please wait a moment and try again."
                else:
                    response = f"⚠️ Media Q&A error: {e}"

    else:
        # Priority 2: RAG / Web / General
        context = ""

        if not st.session_state.force_web_search and st.session_state.rag_enabled:
            docs, _, has_docs = retrieve_documents(
                prompt, chroma_client, COLLECTION_NAME, st.session_state.similarity_threshold
            )
            if has_docs:
                context = "\n\n".join(docs)

        if (st.session_state.force_web_search or not context) and st.session_state.use_web_search:
            with st.spinner("🔍 Searching the web…"):
                try:
                    web_content = get_web_search_agent().run(prompt).content
                    if web_content:
                        context = f"Web information:\n\n{web_content}"
                except Exception as web_err:
                    e = str(web_err)
                    if "429" in e or "quota" in e.lower() or "RESOURCE_EXHAUSTED" in e:
                        st.warning("⚠️ Web search temporarily unavailable — API quota reached.")
                    else:
                        st.warning("⚠️ Web search failed. Answering from general knowledge.")

        with st.spinner("🤖 Generating response…"):
            try:
                full_prompt = ""
                if history_context:
                    full_prompt += f"Previous conversation:\n{history_context}\n\n"
                if context:
                    full_prompt += f"Context:\n{context}\n\n"
                full_prompt += (
                    f"Question:\n{prompt}\n\n"
                    "Answer using the conversation history and context if available, "
                    "otherwise use general knowledge. Stay consistent with prior answers."
                )
                response = get_rag_agent().run(full_prompt).content
            except Exception as gen_err:
                e = str(gen_err)
                if "429" in e or "quota" in e.lower() or "RESOURCE_EXHAUSTED" in e:
                    response = (
                        "⚠️ **API Quota Exceeded**\n\n"
                        "The Gemini free-tier request limit has been reached. "
                        "Please wait a few minutes and try again."
                    )
                else:
                    response = "⚠️ **Error generating response.** Please try again."

    clean_response = filter_think_tags(response)
    st.session_state.history.append({"role": "assistant", "content": clean_response})
    st.session_state.last_response = clean_response

    # Pre-generate downloads
    ts = datetime.now().strftime("%H%M%S")
    st.session_state.dl_timestamp = ts
    try:
        st.session_state.dl_pdf_bytes  = generate_full_conversation_pdf(st.session_state.history)
        st.session_state.dl_pdf_error  = None
    except Exception as e:
        st.session_state.dl_pdf_bytes  = None
        st.session_state.dl_pdf_error  = str(e)
    try:
        st.session_state.dl_audio_bytes = generate_full_conversation_audio(st.session_state.history)
        st.session_state.dl_audio_error = None
    except Exception as e:
        st.session_state.dl_audio_bytes = None
        st.session_state.dl_audio_error = str(e)

    _save_current_conversation()
    st.rerun()

else:
    if not st.session_state.history:
        st.info("💬 Type a question or click a suggestion above.")


# ══════════════════════════════════════════════
# PERSISTENT DOWNLOAD SECTION
# ══════════════════════════════════════════════
if st.session_state.last_response:
    st.markdown(
        "<div class='download-card'>"
        "<h5>📥 Download Full Conversation</h5>"
        "</div>",
        unsafe_allow_html=True,
    )

    q_count = len([m for m in st.session_state.history if m["role"] == "user"])
    st.caption(f"Includes all {q_count} question(s) and answer(s) in this session.")

    dl1, dl2 = st.columns(2)
    ts = st.session_state.dl_timestamp or datetime.now().strftime("%H%M%S")

    with dl1:
        if st.session_state.dl_pdf_bytes:
            st.download_button(
                label="⬇️ Download as PDF",
                data=st.session_state.dl_pdf_bytes,
                file_name=f"conversation_{ts}.pdf",
                mime="application/pdf",
                key="full_conv_pdf",
                use_container_width=True,
            )
        else:
            st.error(f"PDF error: {st.session_state.dl_pdf_error}")

    with dl2:
        if st.session_state.dl_audio_bytes:
            st.download_button(
                label="🔊 Download as Audio",
                data=st.session_state.dl_audio_bytes,
                file_name=f"conversation_{ts}.mp3",
                mime="audio/mpeg",
                key="full_conv_audio",
                use_container_width=True,
            )
        else:
            st.error(f"Audio error: {st.session_state.dl_audio_error}")


# ══════════════════════════════════════════════
# QUIZ SECTION
# ══════════════════════════════════════════════
if st.session_state.quiz_data:
    st.divider()
    st.subheader("🧠 Quiz")
    total = len(st.session_state.quiz_data)

    for i, q in enumerate(st.session_state.quiz_data):
        st.markdown(f"**Q{i+1}. {q['question']}**")
        ans = st.radio(
            "Select your answer:", q["options"], index=None,
            key=f"quiz_q{i}", disabled=st.session_state.quiz_submitted,
        )
        if ans is not None:
            st.session_state.quiz_answers[i] = ans
        st.markdown("")

    answered = len(st.session_state.quiz_answers)
    st.progress(answered / total)
    st.caption(f"Answered {answered} of {total} questions")

    if not st.session_state.quiz_submitted:
        if st.button("✅ Submit Quiz"):
            if answered < total:
                st.warning("Please answer all questions.")
            else:
                st.session_state.quiz_submitted = True
                st.rerun()

    if st.session_state.quiz_submitted:
        score = 0
        st.subheader("📊 Quiz Results")
        for i, q in enumerate(st.session_state.quiz_data):
            user_ans   = st.session_state.quiz_answers.get(i)
            st.markdown(f"**Q{i+1}. {q['question']}**")
            correct    = q["answer"].strip()
            user_check = (user_ans or "").strip()
            if user_check == correct or user_check.startswith(correct):
                st.success(f"✅ Correct — {user_ans}")
                score += 1
            else:
                st.error(f"❌ Your Answer: {user_ans}")
                st.info(f"**Correct Answer:** {q['answer']}")
                st.info(f"**Explanation:** {q['explanation']}")
            st.markdown("")
        st.markdown(f"### 🏆 Score: {score}/{total}")
