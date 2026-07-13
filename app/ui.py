from __future__ import annotations

import sys
import os
from pathlib import Path

import streamlit as st

# Streamlit executes this file with app/ as sys.path[0]. Add the project root so
# the documented `streamlit run app/ui.py` command works from any directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_model_config
from app.inference import OOSDetector, check_vllm_server

st.set_page_config(page_title="机器狗拒识模型测试", page_icon="🐕")
st.title("机器狗拒识模型测试")

config = get_model_config(os.getenv("ROVER_MODEL_KEY") or None)
st.caption(f"当前模型：{config.display_name} · vLLM：{config.base_url}")
server_ready, server_message = check_vllm_server(config)
if server_ready:
    st.success(server_message, icon="✅")
else:
    st.error(f"{server_message}。请先启动对应模型服务。", icon="🚨")

prompt_label = st.radio("提示词", ["Few-shot", "Zero-shot"], horizontal=True)
prompt_name = "zero_shot" if prompt_label == "Zero-shot" else "few_shot"

if "history" not in st.session_state:
    st.session_state.history = []

examples = ["向前走两米", "你现在还有多少电", "你会不会跳舞", "帮我拿桌上的杯子", "帮我订一张机票"]
selected_example = st.selectbox("快速测试示例", ["（请选择）", *examples])
if selected_example != "（请选择）":
    st.session_state.input_text = selected_example

text = st.text_area("用户文本", key="input_text", placeholder="请输入要检测的请求", height=120)
if st.button("检测", type="primary", width="stretch"):
    server_ready, server_message = check_vllm_server(config)
    if not server_ready:
        st.error(f"检测前服务检查失败：{server_message}")
        st.stop()
    result = OOSDetector(prompt_name=prompt_name, model_config=config).predict(text)
    history_item = result.model_dump()
    history_item["prompt_name"] = prompt_name
    st.session_state.history.insert(0, history_item)
    if result.error:
        st.error(result.error)
    if result.prediction == "ACCEPT":
        st.success("ACCEPT")
    elif result.prediction == "REJECT":
        st.warning("REJECT")
    else:
        st.error("INVALID")
    col1, col2 = st.columns(2)
    col1.metric("推理耗时", f"{result.latency_ms:.1f} ms")
    col2.metric("模型", result.model_name)
    st.text_area("原始输出", result.raw_output, disabled=True)

st.subheader("最近测试记录")
if st.session_state.history:
    st.dataframe(
        [{"文本": row["text"], "结果": row["prediction"], "提示词": row.get("prompt_name", "未知"),
          "模型": row["model_name"], "原始输出": row["raw_output"], "耗时 (ms)": round(row["latency_ms"], 1),
          "错误": row["error"] or ""} for row in st.session_state.history[:20]],
        width="stretch",
        hide_index=True,
    )
else:
    st.caption("暂无记录")
