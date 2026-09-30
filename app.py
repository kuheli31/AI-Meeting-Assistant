"""
AI Video Assistant - Streamlit UI
Run with:  streamlit run app.py
"""

import os
import tempfile
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_actionable_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# ----------------------------------------------------------------------------
# Page config & styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .hero {
        padding: 1.6rem 2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
    }
    .hero h1 { margin: 0; font-size: 2rem; color: white; }
    .hero p  { margin: .3rem 0 0 0; opacity: .9; }

    .stat-card {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,.25);
        background: rgba(128,128,128,.06);
        text-align: center;
    }
    .stat-card .num  { font-size: 1.6rem; font-weight: 700; }
    .stat-card .lbl  { font-size: .8rem; opacity: .7; text-transform: uppercase; letter-spacing: .05em; }

    .content-card {
        padding: 1.2rem 1.4rem;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,.25);
        background: rgba(128,128,128,.05);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 8px 16px;
    }
    div.stButton > button:first-child {
        border-radius: 10px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
DEFAULTS = {
    "result": None,
    "messages": [],
    "processed_at": None,
    "pending_question": None,
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def to_markdown(content) -> str:
    """Turn whatever the extractors return (str / list / dict) into markdown."""
    if content is None or content == "" or content == []:
        return "_Nothing found._"
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "\n".join(f"- {item}" for item in content)
    if isinstance(content, dict):
        return "\n".join(f"- **{k}**: {v}" for k, v in content.items())
    return str(content)


def count_items(content) -> int:
    if not content:
        return 0
    if isinstance(content, (list, tuple, dict)):
        return len(content)
    lines = [l for l in str(content).splitlines() if l.strip()]
    bullets = [l for l in lines if l.lstrip().startswith(("-", "*", "•")) or l.lstrip()[:2].rstrip(".").isdigit()]
    return len(bullets) or len(lines)


def build_report(res: dict) -> str:
    return (
        f"# {res['title']}\n\n"
        f"_Generated on {datetime.now():%Y-%m-%d %H:%M}_\n\n"
        f"## Summary\n{to_markdown(res['summary'])}\n\n"
        f"## Action Items\n{to_markdown(res['action_items'])}\n\n"
        f"## Key Decisions\n{to_markdown(res['key_decisions'])}\n\n"
        f"## Open Questions\n{to_markdown(res['open_questions'])}\n\n"
        f"## Transcript\n{res['transcript']}\n"
    )


def save_upload(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


def run_pipeline_ui(source: str, language: str) -> dict:
    """Same steps as run_pipeline(), but with live progress in the UI."""
    with st.status("Processing your video...", expanded=True) as status:
        progress = st.progress(0)

        st.write("🎧 Downloading / splitting audio...")
        chunks = process_input(source)
        progress.progress(15)

        st.write(f"📝 Transcribing ({len(chunks)} chunk(s))...")
        transcript = transcribe_all(chunks, language)
        progress.progress(45)

        st.write("🏷️ Generating title...")
        title = generate_title(transcript)
        progress.progress(55)

        st.write("📋 Summarizing...")
        summary = summarize(transcript)
        progress.progress(68)

        st.write("✅ Extracting action items, decisions & questions...")
        action_items = extract_actionable_items(transcript)
        decisions = extract_key_decisions(transcript)
        questions = extract_questions(transcript)
        progress.progress(88)

        st.write("🧠 Building chat index (RAG)...")
        rag_chain = build_rag_chain(transcript)
        progress.progress(100)

        status.update(label="Done! Your video is ready.", state="complete", expanded=False)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎬 AI Video Assistant")
    st.caption("Turn any video or meeting into notes you can chat with.")
    st.divider()

    input_mode = st.radio(
        "Input type",
        ["🔗 YouTube URL", "📁 Upload file"],
        horizontal=True,
    )

    source = None
    if input_mode == "🔗 YouTube URL":
        url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
        source = url.strip() or None
    else:
        uploaded = st.file_uploader(
            "Audio / video file",
            type=["mp3", "wav", "m4a", "mp4", "mkv", "mov", "webm", "ogg", "flac"],
        )
        if uploaded:
            source = ("upload", uploaded)

    language = st.selectbox("Language", ["english", "hinglish"], index=0)

    go = st.button("🚀 Analyze", type="primary", use_container_width=True, disabled=source is None)

    if st.session_state.result:
        st.divider()
        if st.button("🗑️ Clear & start over", use_container_width=True):
            st.session_state.result = None
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.caption("Tip: longer videos take longer to transcribe.")

# ----------------------------------------------------------------------------
# Run pipeline
# ----------------------------------------------------------------------------
if go and source:
    try:
        path_or_url = save_upload(source[1]) if isinstance(source, tuple) else source
        st.session_state.result = run_pipeline_ui(path_or_url, language)
        st.session_state.messages = []
        st.session_state.processed_at = datetime.now()
    except Exception as e:
        st.error(f"Something went wrong: {e}")
        with st.expander("Details"):
            st.exception(e)

# ----------------------------------------------------------------------------
# Main area
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🎬 AI Video Assistant</h1>
        <p>Transcribe, summarize, extract action items — then chat with your video.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

res = st.session_state.result

if not res:
    c1, c2, c3 = st.columns(3)
    c1.info("**1. Add a source**\n\nPaste a YouTube link or upload a recording in the sidebar.")
    c2.info("**2. Analyze**\n\nGet a transcript, summary, action items, decisions and open questions.")
    c3.info("**3. Chat**\n\nAsk anything about the content and get grounded answers.")
    st.stop()

# ---- Title + stats ----
st.subheader(f"📌 {res['title']}")

words = len(res["transcript"].split())
s1, s2, s3, s4 = st.columns(4)
for col, num, lbl in [
    (s1, f"{words:,}", "Words"),
    (s2, count_items(res["action_items"]), "Action items"),
    (s3, count_items(res["key_decisions"]), "Decisions"),
    (s4, count_items(res["open_questions"]), "Open questions"),
]:
    col.markdown(
        f'<div class="stat-card"><div class="num">{num}</div><div class="lbl">{lbl}</div></div>',
        unsafe_allow_html=True,
    )

st.write("")

# ---- Downloads ----
d1, d2, _ = st.columns([1, 1, 3])
d1.download_button(
    "⬇️ Full report (.md)",
    data=build_report(res),
    file_name="video_report.md",
    mime="text/markdown",
    use_container_width=True,
)
d2.download_button(
    "⬇️ Transcript (.txt)",
    data=res["transcript"],
    file_name="transcript.txt",
    mime="text/plain",
    use_container_width=True,
)

# ---- Tabs ----
tab_chat, tab_sum, tab_act, tab_dec, tab_q, tab_tr = st.tabs(
    ["💬 Chat", "📋 Summary", "✅ Action Items", "🔑 Decisions", "❓ Open Questions", "📝 Transcript"]
)

with tab_sum:
    st.markdown(f'<div class="content-card">{to_markdown(res["summary"])}</div>', unsafe_allow_html=True)

with tab_act:
    st.markdown(to_markdown(res["action_items"]))

with tab_dec:
    st.markdown(to_markdown(res["key_decisions"]))

with tab_q:
    st.markdown(to_markdown(res["open_questions"]))

with tab_tr:
    query = st.text_input("🔍 Search transcript", placeholder="Type a keyword...")
    text = res["transcript"]
    if query:
        hits = [
            line for line in text.replace("\n", " ").split(". ")
            if query.lower() in line.lower()
        ]
        st.caption(f"{len(hits)} matching sentence(s)")
        for h in hits:
            st.markdown(f"- {h.strip()}.")
    else:
        st.text_area("Full transcript", text, height=450, label_visibility="collapsed")

with tab_chat:
    # Suggested questions
    if not st.session_state.messages:
        st.caption("Try one of these:")
        suggestions = [
            "What are the main takeaways?",
            "Who is responsible for what?",
            "What deadlines were mentioned?",
            "What issues remain unresolved?",
        ]
        cols = st.columns(len(suggestions))
        for col, s in zip(cols, suggestions):
            if col.button(s, use_container_width=True):
                st.session_state.pending_question = s
                st.rerun()

    # History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

    # Input (typed or from a suggestion button)
    typed = st.chat_input("Ask anything about this video...")
    question = typed or st.session_state.pending_question
    st.session_state.pending_question = None

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    answer = ask_question(res["rag_chain"], question)
                except Exception as e:
                    answer = f"⚠️ Sorry, I hit an error: {e}"
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.session_state.messages:
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()