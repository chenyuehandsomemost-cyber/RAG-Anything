import streamlit as st
import sys
import os
import asyncio
import re
from pathlib import Path

# === 1. Windows 异步策略补丁 ===
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# === 2. 基础环境配置 ===
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.append(os.getcwd())

from sentence_transformers import SentenceTransformer
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from raganything import RAGAnything, RAGAnythingConfig
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

st.set_page_config(page_title="新工科 AI 助教", layout="wide", page_icon="🎓")

# === 3. 永久事件循环管理 ===
if "loop" not in st.session_state:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    st.session_state.loop = loop
else:
    asyncio.set_event_loop(st.session_state.loop)

# === 4. 辅助函数：清洗数学公式格式 ===
def process_math_format(text):
    if not isinstance(text, str): return str(text)
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    def remove_code_ticks(match):
        content = match.group(1)
        if '\\' in content or '=' in content or '^' in content:
            return f"${content.strip('$')}$"
        return match.group(0)
    text = re.sub(r'`(.*?)`', remove_code_ticks, text)
    return text

# === 5. 模型加载 ===
@st.cache_resource
def load_local_model_only():
    print("正在加载本地 BGE-Small 中文模型...")
    return SentenceTransformer('BAAI/bge-small-zh-v1.5')

# === 6. 核心 RAG 业务逻辑 ===
async def run_rag(file_path, query, level):
    api_key = os.getenv("LLM_BINDING_API_KEY")
    base_url = os.getenv("LLM_BINDING_HOST")
    
    local_model = load_local_model_only()

    async def _current_loop_embed(texts):
        return await asyncio.to_thread(lambda: local_model.encode(texts))

    embedding_func = EmbeddingFunc(
        embedding_dim=512, 
        max_token_size=512, 
        func=_current_loop_embed
    )

    # 普适性增强 Prompt
    query_suffix = ""
    if "初学者" in level:
        query_suffix = """\n\n【指令：直觉科普模式】
        1. 🚫 严禁使用晦涩专业术语，必须用大白话。
        2. ✅ 核心：使用生活中的类比（如把电路比作水管）。
        3. 语气：幽默风趣的科普博主。
        """
    elif "专家" in level:
        query_suffix = """\n\n【指令：深度研讨模式】
        1. ⚠️ 跳过基础定义，假设用户是同行。
        2. ✅ 核心：切入问题本质、底层机制、局限性。
        3. 语气：极度简练、学术、高冷。
        """
    else:
        query_suffix = """\n\n【指令：标准教学模式】
        1. 目标：帮助通过期末考试。
        2. ✅ 结构：定义 -> 公式 -> 物理意义 -> 考点。
        3. 语气：耐心的大学助教。
        """

    async def safe_deepseek_call(prompt, system_prompt="You are a helpful AI tutor.", history_messages=[], **kwargs):
        if 'response_format' in kwargs: del kwargs['response_format']
        
        # 清洗图片消息
        if "messages" in kwargs:
            clean_msgs = []
            for msg in kwargs["messages"]:
                content = msg.get("content")
                if isinstance(content, list):
                    text_content = "".join([item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"])
                    clean_msgs.append({"role": msg["role"], "content": text_content})
                else:
                    clean_msgs.append(msg)
            kwargs["messages"] = clean_msgs

        response = await openai_complete_if_cache(
            "deepseek-chat", prompt, system_prompt=system_prompt, 
            history_messages=history_messages, api_key=api_key, base_url=base_url, **kwargs
        )
        raw_text = response.replace("```json", "").replace("```", "").strip() if isinstance(response, str) else str(response)
        return process_math_format(raw_text)

    async def vision_func(prompt, **kwargs): return await safe_deepseek_call(prompt, **kwargs)

    rag = RAGAnything(
        config=RAGAnythingConfig(working_dir="./rag_storage", parser="mineru", parse_method="auto"),
        llm_model_func=safe_deepseek_call,
        vision_model_func=vision_func,
        embedding_func=embedding_func
    )

    if file_path:
        await rag.process_document_complete(file_path=file_path, output_dir="./output", parse_method="auto")

    return await rag.aquery(query + query_suffix, mode="hybrid")

# === 7. 界面 UI 构建 ===
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=60)
    st.title("⚙️ 学习设置")
    user_level = st.radio("我是谁？", ["👶 初学者 (通俗易懂)", "👨‍🎓 本科生 (专业推导)", "👨‍🔬 领域专家 (深度研讨)"], index=1)
    st.divider()
    
    st.header("📂 知识库")
    uploaded_file = st.file_uploader("上传教材 (PDF)", type=["pdf"])
    
    if st.button("🗑️ 清空知识库缓存"):
        import shutil
        if os.path.exists("./rag_storage"): shutil.rmtree("./rag_storage")
        if os.path.exists("./output"): shutil.rmtree("./output")
        st.success("缓存已清空！请重新上传文件。")

# === 关键修复：在这里全局处理 file_path，确保无论怎么触发都能拿到路径 ===
file_path = None
if uploaded_file:
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", uploaded_file.name)
    # 避免重复写入
    if not os.path.exists(file_path):
        with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
elif os.path.exists("uploads") and len(os.listdir("uploads")) > 0:
    # 如果没重新上传，但文件夹里有旧文件，也自动读取
    file_path = os.path.join("uploads", os.listdir("uploads")[0])

# === 主界面内容 ===
st.title("🎓 新工科 AI 助教系统")
st.caption(f"当前模式：{user_level} | 引擎：DeepSeek-V3 + BGE-Small")

if "messages" not in st.session_state: st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

# --- 按钮回调函数 ---
def click_quiz_btn():
    st.session_state.messages.append({
        "role": "user", 
        "content": "请根据当前文档内容，出 3 道单项选择题，考察核心概念，并附带答案解析。"
    })

col1, col2 = st.columns(2)
with col1:
    # 绑定回调
    st.button("📝 生成随堂测验 (3题)", on_click=click_quiz_btn)

# 处理输入框
if prompt := st.chat_input("请输入你的问题（支持专业公式询问）..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# --- 统一应答逻辑 (核心修复点) ---
# 只要最新一条消息是用户发的，就开始处理
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_user_query = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        with st.spinner("🧠 DeepSeek 正在思考..."):
            try:
                # 检查 file_path 是否有效 (这里 file_path 是全局变量，肯定能访问到)
                if not file_path and not os.path.exists("./rag_storage"):
                    error_msg = "请先在左侧上传 PDF 教材！"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                else:
                    loop = st.session_state.loop
                    # 调用 RAG
                    response = loop.run_until_complete(run_rag(file_path, last_user_query, user_level))
                    
                    import json
                    try:
                        if isinstance(response, str):
                            final_ans = json.loads(response).get("answer", response)
                        else:
                            final_ans = str(response)
                    except:
                        final_ans = str(response)
                    
                    final_ans = process_math_format(final_ans)
                    
                    st.markdown(final_ans)
                    st.session_state.messages.append({"role": "assistant", "content": final_ans})
            except Exception as e:
                st.error(f"发生错误: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})