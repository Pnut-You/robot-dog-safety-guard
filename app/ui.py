from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# Streamlit executes this file with app/ as sys.path[0]. Add the project root so
# the documented `streamlit run app/ui.py` command works from any directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_model_config
from app.inference import OOSDetector, check_vllm_server

st.set_page_config(
    page_title="Rover OOS Detection",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    :root {
        --rover-border: #dbe4f0;
        --rover-muted: #64748b;
        --rover-surface: #ffffff;
        --rover-soft: #f6f9fd;
        --rover-navy: #0b1f3a;
        --rover-navy-light: #173b69;
        --rover-green: #15803d;
        --rover-green-soft: #f0fdf4;
        --rover-red: #dc2626;
        --rover-red-soft: #fef2f2;
        --rover-blue: #2563eb;
        --rover-blue-soft: #dbeafe;
    }
    .stApp {
        background:
            radial-gradient(circle at 88% 4%, rgba(37, 99, 235, .08), transparent 25rem),
            #f4f7fb;
        color: #172033;
    }
    header[data-testid="stHeader"],
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"],
    .stDeployButton {
        display: none !important;
    }
    section[data-testid="stMain"] {
        padding-top: 0 !important;
    }
    .block-container {
        max-width: 1360px;
        padding: 0 clamp(1rem, 3vw, 2.5rem) 3rem;
        margin: 0 auto;
    }
    .rover-navbar {
        position: relative;
        z-index: 10;
        width: 100vw;
        min-height: 76px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.5rem;
        padding: .9rem max(clamp(1rem, 3vw, 2.5rem), calc((100vw - 1360px) / 2 + 2.5rem));
        margin: 0 0 1.5rem calc(50% - 50vw);
        border: 0;
        border-radius: 0;
        background: linear-gradient(110deg, var(--rover-navy) 0%, #102d51 58%, var(--rover-navy-light) 100%);
        box-shadow: 0 8px 28px rgba(11, 31, 58, .18);
    }
    .rover-navbar-title {
        flex: 1 1 auto;
        color: #ffffff;
        font-size: clamp(1.3rem, 2.2vw, 1.72rem);
        font-weight: 750;
        letter-spacing: -.025em;
        white-space: nowrap;
        text-align: left;
    }
    .rover-navbar-meta {
        flex: 0 1 auto;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: .55rem;
        color: #dbeafe;
        font-size: .9rem;
        flex-wrap: wrap;
    }
    .rover-model-chip, .rover-health-chip {
        display: inline-flex;
        align-items: center;
        min-height: 34px;
        padding: .35rem .78rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, .1);
        border: 1px solid rgba(255, 255, 255, .2);
        backdrop-filter: blur(8px);
    }
    .rover-health-chip::before {
        content: "";
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #22c55e;
        margin-right: .4rem;
    }
    .rover-health-chip.offline::before { background: #ef4444; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--rover-surface);
        border-color: var(--rover-border);
        border-radius: 16px;
        box-shadow: 0 8px 28px rgba(15, 40, 72, .055);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: clamp(1rem, 2vw, 1.35rem);
    }
    .stButton > button {
        min-height: 2.75rem;
        border-radius: 10px;
        font-size: .96rem;
        font-weight: 600;
        white-space: normal;
    }
    button[data-testid="stBaseButton-primary"] {
        color: #ffffff;
        background: linear-gradient(100deg, #1d4ed8, #2563eb);
        border: 1px solid #2563eb;
        box-shadow: 0 5px 14px rgba(37, 99, 235, .18);
    }
    button[data-testid="stBaseButton-primary"]:hover {
        color: #ffffff;
        background: linear-gradient(100deg, #1e40af, #1d4ed8);
        border-color: #1d4ed8;
    }
    button[data-testid="stBaseButton-primary"]:disabled {
        color: #94a3b8;
        background: #f1f5f9;
        border-color: #e2e8f0;
    }
    h2 { font-size: 1.42rem !important; letter-spacing: -.02em; }
    h3 { font-size: 1.14rem !important; letter-spacing: -.01em; }
    .rover-status-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: clamp(.8rem, 1.7vw, 1.2rem);
        margin: .85rem 0 1rem;
    }
    .rover-route {
        min-height: 112px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        border: 2px solid var(--rover-border);
        border-radius: 14px;
        background: var(--rover-soft);
        color: #94a3b8;
        transition: all .2s ease;
    }
    .rover-route strong { font-size: clamp(1.35rem, 2.2vw, 1.78rem); letter-spacing: .06em; }
    .rover-route span { font-size: .84rem; margin-top: .32rem; }
    .rover-route.accept-active {
        color: var(--rover-green);
        border-color: #22c55e;
        background: var(--rover-green-soft);
        box-shadow: 0 0 0 4px rgba(34, 197, 94, .10);
    }
    .rover-route.reject-active {
        color: var(--rover-red);
        border-color: #ef4444;
        background: var(--rover-red-soft);
        box-shadow: 0 0 0 4px rgba(239, 68, 68, .10);
    }
    @media (max-width: 700px) {
        .block-container { padding: 0 .75rem 1.75rem; }
        .rover-navbar {
            min-height: 0;
            padding: .9rem 1rem;
            margin-bottom: 1rem;
            align-items: flex-start;
            flex-direction: column;
            gap: .6rem;
        }
        .rover-navbar-title { width: 100%; white-space: normal; }
        .rover-navbar-meta { justify-content: flex-start; }
        .rover-status-grid { grid-template-columns: 1fr; }
        div[data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
        div[data-testid="column"] { min-width: min(100%, 220px) !important; flex: 1 1 220px !important; }
    }
</style>
""",
    unsafe_allow_html=True,
)

config = get_model_config(os.getenv("ROVER_MODEL_KEY") or None)
server_ready, server_message = check_vllm_server(config)
health_class = "" if server_ready else "offline"
health_text = "服务在线" if server_ready else "服务离线"
st.markdown(
    f"""
<nav class="rover-navbar" data-testid="rover-navbar">
  <div class="rover-navbar-title">Rover OOS Detection</div>
  <div class="rover-navbar-meta">
    <span class="rover-model-chip">{config.display_name}</span>
    <span class="rover-health-chip {health_class}">{health_text}</span>
  </div>
</nav>
""",
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

if not server_ready:
    st.error(f"{server_message}。请先启动对应模型服务。", icon="🚨")

examples = [
    ("向前走两米", "action"),
    ("你现在还有多少电", "qa"),
    ("你会不会跳舞", "ability"),
    ("帮我拿桌上的杯子", "near_oos"),
    ("帮我订一张机票", "far_oos"),
]

with st.container(border=True):
    st.subheader("快速测试示例")
    st.caption("点击示例会填入输入框，再点击检测按钮发起判断。")
    example_columns = st.columns(len(examples))
    for column, (example, key) in zip(example_columns, examples, strict=True):
        if column.button(example, key=f"example_{key}", width="stretch"):
            st.session_state.input_text = example

with st.container(border=True):
    st.subheader("请求检测")
    text = st.text_area(
        "用户文本",
        key="input_text",
        placeholder="例如：向前走两米",
        height=130,
        label_visibility="collapsed",
    )
    detect_clicked = st.button("开始检测", type="primary", width="stretch", disabled=not server_ready)

    if detect_clicked:
        server_ready, server_message = check_vllm_server(config)
        if not server_ready:
            st.session_state.latest_result = None
            st.error(f"检测前服务检查失败：{server_message}")
        else:
            result = OOSDetector(prompt_name="few_shot", model_config=config).predict(text)
            history_item = result.model_dump()
            st.session_state.latest_result = history_item
            st.session_state.history.insert(0, history_item)

    latest = st.session_state.latest_result
    prediction = latest["prediction"] if latest else None
    accept_class = "accept-active" if prediction == "ACCEPT" else ""
    reject_class = "reject-active" if prediction == "REJECT" else ""
    st.markdown(
        f"""
<div class="rover-status-grid" data-testid="route-result">
  <div class="rover-route {accept_class}"><strong>ACCEPT</strong><span>能力范围内且安全</span></div>
  <div class="rover-route {reject_class}"><strong>REJECT</strong><span>超出能力或存在风险</span></div>
</div>
""",
        unsafe_allow_html=True,
    )

    if latest:
        if latest["error"]:
            st.error(latest["error"])
        elif prediction == "INVALID":
            st.error("模型输出不是完整的 ACCEPT 或 REJECT，本次结果标记为 INVALID。")
        detail_col, latency_col, raw_col = st.columns([1, 1, 2])
        detail_col.metric("检测结果", prediction)
        latency_col.metric("推理耗时", f"{latest['latency_ms']:.1f} ms")
        raw_col.text_input("原始输出", latest["raw_output"], disabled=True)

st.subheader("最近测试记录")
if st.session_state.history:
    st.dataframe(
        [
            {
                "文本": row["text"],
                "结果": row["prediction"],
                "模型": row["model_name"],
                "原始输出": row["raw_output"],
                "耗时 (ms)": round(row["latency_ms"], 1),
                "错误": row["error"] or "",
            }
            for row in st.session_state.history[:20]
        ],
        width="stretch",
        hide_index=True,
    )
else:
    st.caption("暂无记录，选择上方示例或输入请求开始检测。")
