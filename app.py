"""
Visistant: Conversational NL-to-Visualization Chatbot
Stack: Streamlit + Google Gemini API (direct) + Manual Buffer Window Memory (k=3) + Plotly
Paper: "Visistant: A Conversational Chatbot for Natural Language to Visualizations With Gemini LLMs"
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import io
import traceback
from PIL import Image
import google.generativeai as genai

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Visistant", page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp{background-color:#0f1117;color:#e0e0e0}
section[data-testid="stSidebar"]{background-color:#161b27;border-right:1px solid #2a2f3e}
.hdr{background:linear-gradient(135deg,#1a1f35,#0d1b2a);border:1px solid #2a3a5c;
     border-radius:12px;padding:18px 24px;margin-bottom:20px}
.hdr h1{font-size:1.8rem;font-weight:700;color:#4fc3f7;margin:0}
.hdr p{color:#8899aa;margin:4px 0 0;font-size:.85rem}
.ub{background:linear-gradient(135deg,#1565c0,#0d47a1);border-radius:18px 18px 4px 18px;
    padding:12px 16px;margin:6px 0;max-width:75%;margin-left:auto;color:#fff;font-size:.92rem}
.ab{background:linear-gradient(135deg,#1a2744,#162035);border:1px solid #2a3a5c;
    border-radius:18px 18px 18px 4px;padding:12px 16px;margin:6px 0;
    max-width:90%;color:#d0d8e8;font-size:.92rem}
.ul{text-align:right;color:#78a9d8;font-size:.75rem;font-weight:600;margin-bottom:2px}
.al{color:#4fc3f7;font-size:.75rem;font-weight:600;margin-bottom:2px}
.ins{background:#0d2a1f;border:1px solid #1b5e35;border-left:4px solid #4caf50;
     border-radius:8px;padding:12px 16px;margin-top:10px;font-size:.88rem;color:#a5d6b0}
.err{background:#2a0a0a;border:1px solid #c62828;border-left:4px solid #f44336;
     border-radius:8px;padding:12px 16px;color:#ef9a9a;font-size:.88rem;margin-top:8px}
.badge{display:inline-block;background:#1e3a5c;color:#4fc3f7;border:1px solid #2a5a8c;
       border-radius:20px;padding:2px 10px;font-size:.72rem;font-weight:600;margin:2px}
.stButton>button{background:linear-gradient(135deg,#1565c0,#0d47a1);
                  color:#fff;border:none;border-radius:8px;font-weight:600}
hr{border-color:#2a3a5c}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def clean_df(df):
    df = df.drop_duplicates()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            m = df[col].mode()
            df[col] = df[col].fillna(m[0] if not m.empty else "Unknown")
    return df


def make_initial_prompt(df, cols=None):
    """Paper §III-B: schema prompt so LLM knows df exists."""
    use_cols = cols if cols else df.columns.tolist()
    lines = ["There is already a pandas DataFrame called `df` with these columns:\n"]
    for c in use_cols:
        if c not in df.columns:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            lines.append(f'  • "{c}" — numeric ({s.dtype})')
        else:
            top20 = s.value_counts().index.tolist()[:20]
            lines.append(f'  • "{c}" — categorical, values: {top20}')
    lines += [
        "\nStrict rules for your response:",
        "1. Use ONLY plotly.express (as px) or plotly.graph_objects (as go).",
        "2. The DataFrame `df` already exists — do NOT import or recreate it.",
        "3. Do NOT call fig.show().",
        "4. Always add a title and axis labels.",
        "5. Return ONLY a Python code block (```python ... ```) — no explanation.",
    ]
    return "\n".join(lines)


def build_prompt(init_prompt, window, query):
    """Paper §III-C: prepend schema + last-3 turns + current query."""
    parts = [init_prompt, "\n--- Conversation History (last 3 turns) ---"]
    for t in window:
        parts.append(f"Human: {t['query']}")
        parts.append(f"AI Code:\n```python\n{t['code']}\n```")
    parts.append(f"\n--- Current Request ---\nHuman: {query}")
    parts.append("AI (Python Plotly code only):")
    return "\n".join(parts)


def call_gemini(api_key, prompt):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        "gemini-2.5-flash",
        generation_config=genai.types.GenerationConfig(temperature=0.1),
    )
    resp = model.generate_content(prompt)
    return resp.text


def extract_code(text):
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def run_code(code, df):
    code = re.sub(r"\bfig\.show\(\)\s*", "", code)
    lv = {"df": df, "px": px, "go": go, "pd": pd}
    try:
        exec(compile(code, "<gen>", "exec"), {}, lv)
        fig = lv.get("fig")
        return (fig, None) if fig is not None else (None, "No `fig` variable found in generated code.")
    except Exception:
        return None, traceback.format_exc()


def gemini_insights(fig, query, api_key):
    try:
        img = Image.open(io.BytesIO(fig.to_image(format="png", scale=2)))
        genai.configure(api_key=api_key)
        m = genai.GenerativeModel("gemini-1.5-flash")
        r = m.generate_content([
            f"Chart for query: '{query}'. Give 2-4 bullet-point insights about patterns, trends, or outliers.",
            img
        ])
        return r.text
    except Exception as e:
        return f"(Insights unavailable: {e})"


def fid(name):
    return "f_" + re.sub(r"\W", "_", name)


# ─────────────────────────────────────────────────────────────────────────────
# Session State Init
# ─────────────────────────────────────────────────────────────────────────────
for k, v in [("stack", []), ("data", {}), ("history", {}), ("window", {})]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    api_key = st.text_input("Google Gemini API Key", type="password", placeholder="AIza...")

    st.markdown("---")
    st.markdown("## 📂 Upload CSV Files")
    uploaded = st.file_uploader("Upload CSV(s)", type=["csv"], accept_multiple_files=True)

    if uploaded:
        for f in uploaded:
            fid_ = fid(f.name)
            if fid_ not in st.session_state.data:
                df_clean = clean_df(pd.read_csv(f))
                st.session_state.data[fid_]    = {"df": df_clean, "name": f.name}
                st.session_state.history[fid_] = []
                st.session_state.window[fid_]  = []
                if fid_ in st.session_state.stack:
                    st.session_state.stack.remove(fid_)
                st.session_state.stack.insert(0, fid_)

    sel_fid = None
    if st.session_state.stack:
        names   = [st.session_state.data[f]["name"] for f in st.session_state.stack]
        picked  = st.selectbox("Select Dataset", names)
        sel_fid = st.session_state.stack[names.index(picked)]

    st.markdown("---")
    mode      = st.radio("Mode", ["Default", "Advanced"])
    sel_cols  = None
    ins_mode  = False

    if sel_fid:
        df_cur = st.session_state.data[sel_fid]["df"]
        if mode == "Advanced":
            sel_cols = st.multiselect("Columns", df_cur.columns.tolist(),
                                      default=df_cur.columns.tolist()[:3])
        ins_mode = st.toggle("💡 AI Insights", value=False)
        st.markdown("---")
        st.markdown("**Preview:**")
        st.dataframe(df_cur.head(5), use_container_width=True)
        r, c = df_cur.shape
        n = len(df_cur.select_dtypes("number").columns)
        cat = c - n
        st.markdown(f"<span class='badge'>Rows:{r}</span><span class='badge'>Cols:{c}</span>"
                    f"<span class='badge'>Num:{n}</span><span class='badge'>Cat:{cat}</span>",
                    unsafe_allow_html=True)
        st.markdown("---")
        if st.button("🗑️ Clear Chat"):
            st.session_state.history[sel_fid] = []
            st.session_state.window[sel_fid]  = []
            st.rerun()

    st.markdown("<small style='color:#556677'>Visistant · Gemini 1.5 Flash · k=3 window</small>",
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main Area
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='hdr'><h1>📊 Visistant</h1>"
            "<p>Conversational NL-to-Visualization · Gemini · Plotly · Buffer Window Memory</p></div>",
            unsafe_allow_html=True)

if not api_key:
    st.info("👈 Enter your **Google Gemini API Key** in the sidebar.")
    st.stop()
if not sel_fid:
    st.info("👈 Upload a **CSV file** to get started.")
    st.stop()

df_cur   = st.session_state.data[sel_fid]["df"]
fname    = st.session_state.data[sel_fid]["name"]
history  = st.session_state.history[sel_fid]
window   = st.session_state.window[sel_fid]

st.markdown(f"**Dataset:** `{fname}` · `{df_cur.shape[0]}` rows × `{df_cur.shape[1]}` cols · **{mode}** mode")
st.markdown("---")

# ── Chat history ──────────────────────────────────────────────────────────────
for i, msg in enumerate(history):
    if msg["role"] == "user":
        st.markdown(f"<div class='ul'>🧑 You</div><div class='ub'>{msg['content']}</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<div class='al'>🤖 Visistant</div>", unsafe_allow_html=True)
        if msg.get("fig"):
            st.markdown("<div class='ab'>Here is your visualization:</div>", unsafe_allow_html=True)
            st.plotly_chart(msg["fig"], use_container_width=True, key=f"c_{sel_fid}_{i}")
        if msg.get("error"):
            st.markdown(f"<div class='err'>⚠️ {msg['error']}</div>", unsafe_allow_html=True)
        if msg.get("insights"):
            st.markdown(f"<div class='ins'><b>💡 Insights:</b><br>{msg['insights']}</div>",
                        unsafe_allow_html=True)
        if i == len(history) - 1 and msg.get("code"):
            with st.expander("🔍 Generated Plotly Code"):
                st.code(msg["code"], language="python")

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown("---")
with st.form("qf", clear_on_submit=True):
    c1, c2 = st.columns([5, 1])
    with c1:
        q = st.text_input("q", placeholder="e.g. Show total sales by region as a bar chart",
                          label_visibility="collapsed")
    with c2:
        go_btn = st.form_submit_button("Send ▶")

# ── Process ───────────────────────────────────────────────────────────────────
if go_btn and q.strip():
    query = q.strip()
    history.append({"role": "user", "content": query})

    with st.spinner("🤖 Generating visualization..."):
        try:
            init_p  = make_initial_prompt(df_cur, sel_cols)
            prompt  = build_prompt(init_p, window[-3:], query)
            raw     = call_gemini(api_key, prompt)
            code    = extract_code(raw)
            fig, err = run_code(code, df_cur)

            if err:
                ai_msg = {"role":"assistant","fig":None,"code":code,
                          "insights":None,"error":err}
            else:
                # Update window memory (k=3)
                window.append({"query": query, "code": code})
                if len(window) > 3:
                    window.pop(0)
                st.session_state.window[sel_fid] = window

                insights = gemini_insights(fig, query, api_key) if ins_mode else None
                ai_msg = {"role":"assistant","fig":fig,"code":code,
                          "insights":insights,"error":None}

        except Exception as e:
            ai_msg = {"role":"assistant","fig":None,"code":None,
                      "insights":None,"error":str(e)}

    history.append(ai_msg)
    st.session_state.history[sel_fid] = history
    st.rerun()
