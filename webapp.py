import os
import uuid
import json
import html
import base64
import hashlib
import tempfile
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from supabase import create_client
except ImportError:
    create_client = None

load_dotenv()

st.set_page_config(
    page_title="Nexora AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- SETTINGS --------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
APP_URL = os.getenv("APP_URL", "http://localhost:8504").strip().rstrip("/")

CHAT_MODEL = "openai/gpt-oss-120b"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
SYSTEM_PROMPT = (
    "You are Nexora AI, a friendly and accurate study assistant. Explain concepts in "
    "simple student-friendly language. Follow requested exam-answer lengths (2, 5, or "
    "10 marks). Prioritize provided PDF context and uploaded images. Never claim a "
    "source says something it does not. If unsure, say so."
)

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
supabase = None
supabase_setup_error = ""
if SUPABASE_URL and SUPABASE_KEY and create_client:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as exc:
        supabase_setup_error = str(exc)

# -------------------- DARK PURPLE UI --------------------
st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background: #09090f !important; color: #f5f3ff !important;
    }
    [data-testid="stSidebar"] {
        background: #10101b !important; border-right: 1px solid #29243b;
    }
    .block-container { max-width: 1120px; padding-top: 1.4rem; padding-bottom: 5rem; }
    h1,h2,h3,p,label,span { color: #f5f3ff; }
    .muted { color: #a9a0c2 !important; }
    .brand-mark {
        width: 54px; height: 54px; border-radius: 17px; display: flex;
        align-items: center; justify-content: center; margin: 0 auto 14px;
        background: linear-gradient(135deg,#8b5cf6,#5b21b6); font-size: 27px;
        box-shadow: 0 8px 32px #6d3ce833;
    }
    .auth-title { text-align: center; font-size: 2rem; font-weight: 750; letter-spacing: -.04em; }
    .auth-subtitle { text-align: center; color: #aaa3bd; margin-bottom: 1.4rem; }
    .auth-card {
        background: #10101a; border: 1px solid #302a45; border-radius: 22px;
        padding: 28px 30px 22px; max-width: 460px; margin: 1rem auto 0;
    }
    .stTextInput input, .stTextArea textarea {
        background: #151522 !important; color: #fff !important; border-color: #38324e !important;
        border-radius: 10px !important;
    }
    .stButton > button, .stFormSubmitButton > button, .stLinkButton > a {
        border-radius: 11px !important; border: 1px solid #7047eb !important;
        background: #6d3ce8 !important; color: white !important; font-weight: 600 !important;
        min-height: 42px;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover, .stLinkButton > a:hover {
        background: #8258ff !important; color: white !important;
    }
    .auth-google .stButton > button {
        background: #151522 !important; border-color: #39334d !important;
    }
    [data-testid="stChatInput"] {
        background: #11111c; border: 1px solid #6d4aff; border-radius: 18px;
    }
    [data-testid="stFileUploader"] {
        background: #12121d; border: 1px dashed #59448d; border-radius: 14px; padding: .5rem;
    }
    .chat-row { display:flex; width:100%; margin: 1rem 0 1.25rem; box-sizing:border-box; }
    .chat-row.user { justify-content:flex-end; }
    .chat-row.assistant { justify-content:flex-start; }
    .chat-bubble {
        display:flex; align-items:flex-start; gap:.7rem; max-width:82%; padding:.9rem 1rem;
        border:1px solid #302a45; border-radius:18px; background:#11111c; color:#f5f3ff;
        overflow-wrap:anywhere; box-sizing:border-box;
    }
    .chat-row.user .chat-bubble {
        background:#241744; border-color:#6844b5; border-top-right-radius:6px;
    }
    .chat-row.assistant .chat-bubble { border-top-left-radius:6px; }
    .chat-avatar {
        flex:0 0 32px; width:32px; height:32px; display:flex; align-items:center;
        justify-content:center; border-radius:10px; background:#6d3ce8;
    }
    .chat-row.user .chat-avatar { background:#52328b; }
    .chat-content { min-width:0; line-height:1.55; white-space:normal; }
    .sidebar-title { font-size:1.45rem; font-weight:800; letter-spacing:-.04em; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- SESSION STATE --------------------
defaults = {
    "messages": [{"role": "assistant", "content": "Hey! 👋 I'm Nexora AI. Ask a question, upload a PDF, or add an image of your notes."}],
    "chat_id": str(uuid.uuid4()),
    "chat_title": "New chat",
    "access_token": None,
    "refresh_token": None,
    "guest_mode": False,
    "auth_notice": "",
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# -------------------- HELPERS --------------------
def now_iso():
    return datetime.now(timezone.utc).isoformat()

def safe_title(value):
    value = " ".join((value or "").strip().split())
    return value[:60] if value else "New chat"

def get_auth_user():
    if not supabase or not st.session_state.access_token:
        return None
    try:
        supabase.postgrest.auth(st.session_state.access_token)
        return supabase.auth.get_user(st.session_state.access_token).user
    except Exception:
        return None

def db_query(method, **kwargs):
    user = get_auth_user()
    if not supabase or not user:
        return None
    try:
        supabase.postgrest.auth(st.session_state.access_token)
        query = supabase.table("chats")
        if method == "list":
            return query.select("*").eq("user_id", str(user.id)).order("updated_at", desc=True).execute().data
        if method == "get":
            return query.select("*").eq("id", kwargs["id"]).eq("user_id", str(user.id)).execute().data
        if method == "insert":
            return query.insert(kwargs["data"]).execute().data
        if method == "update":
            return query.update(kwargs["data"]).eq("id", kwargs["id"]).eq("user_id", str(user.id)).execute().data
        if method == "delete":
            return query.delete().eq("id", kwargs["id"]).eq("user_id", str(user.id)).execute().data
    except Exception as exc:
        st.sidebar.warning(f"Chat storage issue: {exc}")
        return None

def save_current_chat():
    user = get_auth_user()
    if not user:
        return
    first_user_message = next((m.get("content", "") for m in st.session_state.messages if m.get("role") == "user"), "")
    title = st.session_state.chat_title
    if title == "New chat" and first_user_message:
        title = safe_title(first_user_message)
        st.session_state.chat_title = title
    data = {
        "title": title,
        "messages": st.session_state.messages,
        "updated_at": now_iso(),
    }
    existing = db_query("get", id=st.session_state.chat_id) or []
    if existing:
        db_query("update", id=st.session_state.chat_id, data=data)
    else:
        data.update({
            "id": st.session_state.chat_id,
            "user_id": str(user.id),
            "created_at": now_iso(),
        })
        db_query("insert", data=data)

def new_chat():
    st.session_state.chat_id = str(uuid.uuid4())
    st.session_state.chat_title = "New chat"
    st.session_state.messages = [
        {"role": "assistant", "content": "New chat started. What would you like to study?"}
    ]

def show_message(message):
    role = message.get("role", "assistant")
    is_user = role == "user"
    side = "user" if is_user else "assistant"
    avatar = "🧑" if is_user else "🎓"
    content = html.escape(str(message.get("content", ""))).replace("\n", "<br>")
    st.markdown(
        f'<div class="chat-row {side}"><div class="chat-bubble">'
        f'<div class="chat-avatar">{avatar}</div><div class="chat-content">{content}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

def read_pdf(upload):
    try:
        reader = PdfReader(upload)
        return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception as exc:
        st.warning(f"Could not read this PDF: {exc}")
        return ""

def relevant_pdf_context(question, text, top_k=4):
    if not text.strip():
        return ""
    chunks = [x.strip() for x in text.split("\n\n") if len(x.strip()) > 40]
    if not chunks:
        chunks = [text[:10000]]
    chunks = chunks[:300]
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(chunks + [question])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
        indexes = scores.argsort()[::-1][:top_k]
        selected = [chunks[i] for i in indexes if scores[i] > 0]
        return "\n\n".join(selected)[:12000]
    except Exception:
        return text[:6000]

def accept_auth_session(session):
    if session:
        st.session_state.access_token = session.access_token
        st.session_state.refresh_token = session.refresh_token
        st.session_state.guest_mode = False

# -------------------- SUPABASE OAUTH CALLBACK --------------------
# Streamlit can create a fresh server session after returning from Google.
# Persist Supabase's PKCE verifier briefly so the callback can exchange the code.
def oauth_verifier_file():
    project_hash = hashlib.sha256(SUPABASE_URL.encode("utf-8")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"nexora_oauth_verifier_{project_hash}.txt"

if supabase and "code" in st.query_params:
    oauth_code = st.query_params.get("code")
    if oauth_code:
        try:
            verifier_path = oauth_verifier_file()
            if not verifier_path.exists():
                raise RuntimeError(
                    "The temporary Google login verifier was not found. Please click "
                    "Continue with Google and complete sign-in again."
                )
            verifier = verifier_path.read_text(encoding="utf-8").strip()
            if not verifier:
                raise RuntimeError("The temporary Google login verifier was empty. Please try again.")

            # Supabase Python SDK stores this value in its auth storage during OAuth start.
            storage = getattr(supabase.auth, "_storage", None)
            if storage is None or not hasattr(storage, "set_item"):
                raise RuntimeError(
                    "Your installed Supabase SDK does not expose the expected auth storage. "
                    "Run: py -m pip install --upgrade supabase"
                )
            storage_key = getattr(supabase.auth, "_storage_key", "supabase.auth.token")
            storage.set_item(f"{storage_key}-code-verifier", verifier)

            auth_result = supabase.auth.exchange_code_for_session({"auth_code": str(oauth_code)})
            accept_auth_session(auth_result.session)
            verifier_path.unlink(missing_ok=True)
            st.query_params.clear()
            st.rerun()
        except Exception as exc:
            st.session_state.auth_notice = (
                "Google sign-in could not finish. Please try Google sign-in again. "
                f"Details: {exc}"
            )
            st.query_params.clear()

# -------------------- AUTH SCREEN --------------------
current_user = get_auth_user()
if not current_user and not st.session_state.guest_mode:
    st.markdown('<div style="height:5vh"></div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-mark">🎓</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-title">Welcome to Nexora AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-subtitle">Your personal AI study space</div>', unsafe_allow_html=True)

    left, center, right = st.columns([1, 1.35, 1])
    with center:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        tab_login, tab_signup = st.tabs(["Sign in", "Create account"])

        with tab_login:
            if st.button("🌐  Continue with Google", key="google_login", use_container_width=True):
                if not supabase:
                    st.error("Add valid SUPABASE_URL and SUPABASE_KEY values to your .env first.")
                else:
                    try:
                        oauth = supabase.auth.sign_in_with_oauth({
                            "provider": "google",
                            "options": {"redirect_to": APP_URL},
                        })
                        oauth_url = getattr(getattr(oauth, "url", None), "url", None) or getattr(oauth, "url", None)
                        if oauth_url:
                            storage = getattr(supabase.auth, "_storage", None)
                            storage_key = getattr(supabase.auth, "_storage_key", "supabase.auth.token")
                            verifier = storage.get_item(f"{storage_key}-code-verifier") if storage and hasattr(storage, "get_item") else None
                            if not verifier:
                                st.error("Supabase did not create a PKCE verifier. Upgrade the Supabase package and try again.")
                            else:
                                oauth_verifier_file().write_text(str(verifier), encoding="utf-8")
                                st.link_button("Continue to Google", oauth_url, use_container_width=True)
                                st.info("Click 'Continue to Google' above to finish signing in.")
                        else:
                            st.error("Supabase did not return a Google sign-in URL.")
                    except Exception as exc:
                        st.error(f"Google sign-in setup error: {exc}")

            st.markdown("<p style='text-align:center;color:#8f88a5'>or continue with email</p>", unsafe_allow_html=True)
            with st.form("signin_form"):
                login_email = st.text_input("Email", placeholder="you@example.com")
                login_password = st.text_input("Password", type="password", placeholder="Your password")
                login_submit = st.form_submit_button("Continue with email", use_container_width=True)
            if login_submit:
                if not supabase:
                    st.error("Supabase is not configured. Check your .env settings.")
                elif not login_email.strip() or not login_password:
                    st.error("Enter your email and password.")
                else:
                    try:
                        result = supabase.auth.sign_in_with_password({
                            "email": login_email.strip(),
                            "password": login_password,
                        })
                        accept_auth_session(result.session)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Sign-in failed: {exc}")

        with tab_signup:
            st.markdown("Create your account to save chats and access them later.")
            with st.form("signup_form"):
                signup_email = st.text_input("Email", placeholder="you@example.com", key="signup_email")
                signup_password = st.text_input("Create password", type="password", key="signup_password")
                signup_confirm = st.text_input("Confirm password", type="password", key="signup_confirm")
                signup_submit = st.form_submit_button("Create account", use_container_width=True)
            if signup_submit:
                if not supabase:
                    st.error("Supabase is not configured. Check your .env settings.")
                elif not signup_email.strip() or not signup_password:
                    st.error("Enter an email and password.")
                elif signup_password != signup_confirm:
                    st.error("Passwords do not match.")
                elif len(signup_password) < 8:
                    st.error("Use a password with at least 8 characters.")
                else:
                    try:
                        result = supabase.auth.sign_up({
                            "email": signup_email.strip(),
                            "password": signup_password,
                        })
                        if result.session:
                            accept_auth_session(result.session)
                            st.rerun()
                        else:
                            st.success("Account created. Check your email to confirm, then sign in.")
                    except Exception as exc:
                        st.error(f"Account creation failed: {exc}")

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;color:#8f88a5;margin-top:16px'>Your notes and chats stay yours.</p>", unsafe_allow_html=True)

    if st.session_state.auth_notice:
        st.warning(st.session_state.auth_notice)
        st.session_state.auth_notice = ""

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    _, guest_col, _ = st.columns([1, 1.35, 1])
    with guest_col:
        if st.button("Continue as guest", use_container_width=True):
            st.session_state.guest_mode = True
            st.rerun()
    st.stop()

# -------------------- MAIN APP SIDEBAR --------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">🎓 Nexora AI</div>', unsafe_allow_html=True)
    st.caption("Your personal AI study space")
    st.divider()

    user = get_auth_user()
    if user:
        st.success(f"Signed in: {getattr(user, 'email', 'account')}")
        if st.button("Sign out", use_container_width=True):
            try:
                if supabase:
                    supabase.auth.sign_out()
            except Exception:
                pass
            st.session_state.access_token = None
            st.session_state.refresh_token = None
            st.session_state.guest_mode = False
            st.rerun()
    else:
        st.caption("Guest mode — chats won't be saved to your account.")
        if st.button("Sign in / Create account", use_container_width=True):
            st.session_state.guest_mode = False
            st.rerun()

    st.divider()
    if st.button("＋ New chat", use_container_width=True):
        new_chat()
        st.rerun()

    if user:
        st.markdown("### Chat history")
        saved_chats = db_query("list") or []
        for chat in saved_chats:
            if st.button((chat.get("title") or "Saved chat")[:34], key=f"chat_{chat['id']}", use_container_width=True):
                st.session_state.chat_id = chat["id"]
                st.session_state.chat_title = chat.get("title") or "Saved chat"
                st.session_state.messages = chat.get("messages") or []
                st.rerun()
        with st.expander("Manage current chat"):
            rename_value = st.text_input("Chat title", value=st.session_state.chat_title, key="rename_chat_title")
            if st.button("Rename chat", use_container_width=True):
                st.session_state.chat_title = safe_title(rename_value)
                save_current_chat()
                st.rerun()
            if st.button("Delete current chat", use_container_width=True):
                db_query("delete", id=st.session_state.chat_id)
                new_chat()
                st.rerun()

    st.divider()
    st.markdown("### Study files")
    pdf_file = st.file_uploader("Upload a PDF", type=["pdf"], key="pdf_upload")
    image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"], key="image_upload")
    if st.button("Clear uploaded files", use_container_width=True):
        st.session_state.pop("pdf_upload", None)
        st.session_state.pop("image_upload", None)
        st.rerun()

# -------------------- MAIN CHAT --------------------
if not groq_client:
    st.error("GROQ_API_KEY is missing. Add it to your .env file or deployment secrets.")
    st.stop()

st.title("NexoraAI")
st.caption("Learn faster. Prepare smarter. Ask questions from your notes.")

for msg in st.session_state.messages:
    show_message(msg)

prompt = st.chat_input("Ask Nexora AI anything...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    if st.session_state.chat_title == "New chat":
        st.session_state.chat_title = safe_title(prompt)
    show_message({"role": "user", "content": prompt})

    with st.spinner("Nexora AI is thinking..."):
        context = ""
        if pdf_file:
            pdf_text = read_pdf(pdf_file)
            if pdf_text:
                context = relevant_pdf_context(prompt, pdf_text)
                if not context:
                    context = pdf_text[:6000]

        user_content = prompt
        if context:
            user_content = (
                "Use this extracted PDF context as the main source. If unsupported, say so.\n\n"
                f"PDF CONTEXT:\n{context}\n\nQUESTION:\n{prompt}"
            )

        try:
            if image_file:
                encoded = base64.b64encode(image_file.getvalue()).decode("utf-8")
                mime = image_file.type or "image/png"
                result = groq_client.chat.completions.create(
                    model=VISION_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": [
                            {"type": "text", "text": user_content + "\nInspect the uploaded study image and use it to answer."},
                            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                        ]},
                    ],
                    temperature=0.3,
                    max_tokens=1800,
                )
            else:
                history = [{"role": "system", "content": SYSTEM_PROMPT}]
                for old in st.session_state.messages[-9:-1]:
                    if old.get("role") in ("user", "assistant"):
                        history.append({"role": old["role"], "content": old["content"]})
                history.append({"role": "user", "content": user_content})
                result = groq_client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=history,
                    temperature=0.4,
                    max_tokens=1800,
                )
            answer = result.choices[0].message.content or "I couldn't generate a response. Please try again."
        except Exception as exc:
            answer = f"Sorry, I couldn't answer that right now. Please check the API/model settings. Details: {exc}"

    show_message({"role": "assistant", "content": answer})
    st.session_state.messages.append({"role": "assistant", "content": answer})
    save_current_chat()

st.markdown('<p class="muted" style="text-align:center;margin-top:2rem">Nexora AI • Built for learning</p>', unsafe_allow_html=True)
