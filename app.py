import streamlit as st
import sys
import os
import asyncio
import re
import json
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

# 加载自定义模块
from database import init_database, save_chat_history, update_knowledge_point
from auth import show_login_page, is_logged_in, get_current_user, get_current_user_id, show_user_info_sidebar
from student_profile import show_profile_page, show_mini_profile_card
from analytics import extract_topic_from_question

load_dotenv(dotenv_path=".env", override=False)

# === Streamlit Cloud 支持：从 secrets 读取配置 ===
def get_env_or_secret(key: str, default: str = None):
    value = os.getenv(key)
    if value:
        return value
    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
        if hasattr(st, 'secrets') and 'api_keys' in st.secrets and key in st.secrets['api_keys']:
            return st.secrets['api_keys'][key]
    except:
        pass
    return default

st.set_page_config(page_title="新工科 AI 助教", layout="wide", page_icon="🎓")

# 初始化数据库
init_database()

# === 视觉模型配置 ===
VISION_PROVIDERS = {
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "model": "glm-4v",
        "env_key": "ZHIPU_API_KEY"
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-vl-max",
        "env_key": "QWEN_API_KEY"
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "Qwen/Qwen2-VL-72B-Instruct",
        "env_key": "SILICONFLOW_API_KEY"
    }
}

# === 3. 永久事件循环管理 ===
if "loop" not in st.session_state:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    st.session_state.loop = loop
else:
    asyncio.set_event_loop(st.session_state.loop)

# === 4. 数学公式清洗函数 (防乱码版) ===
def process_math_format(text):
    """
    针对 Streamlit/KaTeX 的清洗函数
    修复：空格导致的不渲染、块级公式不换行、转义符冲突
    """
    if not isinstance(text, str): return str(text)

    # 1. 移除 Markdown 代码块标记
    text = re.sub(r'```latex\s*', '', text)
    text = re.sub(r'```\s*', '', text)

    # 2. 替换定界符
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
    text = re.sub(r'\\\[(.*?)\\\]', r'\n$$\1$$\n', text, flags=re.DOTALL)

    # 3. 去除行内公式 $ 内部首尾的空格
    text = re.sub(r'\$\s+([^$]+?)\s+\$', r'$\1$', text)

    # 4. 确保块级公式 $$ 前后强制换行
    def fix_block_math(match):
        content = match.group(1).strip()
        return f"\n$$\n{content}\n$$\n"
    
    text = re.sub(r'\$\$([\s\S]+?)\$\$', fix_block_math, text)

    # 5. 修复常见的 LaTeX 字符转义错误
    text = text.replace(r'\$', '$')
    text = text.replace(r'\%', '%')

    return text

# === 5. 模型加载 ===
@st.cache_resource
def load_local_model_only():
    # 仅作为 Embedding 使用
    return SentenceTransformer('BAAI/bge-small-zh-v1.5')

# === 6. 核心 RAG 业务逻辑 ===
async def run_rag(file_path, query, level, is_quiz_mode=False):
    api_key = get_env_or_secret("LLM_BINDING_API_KEY")
    base_url = get_env_or_secret("LLM_BINDING_HOST")
    
    vision_provider = get_env_or_secret("VISION_PROVIDER", "zhipu")
    vision_config = VISION_PROVIDERS.get(vision_provider, VISION_PROVIDERS["zhipu"])
    vision_api_key = get_env_or_secret(vision_config["env_key"]) or get_env_or_secret("VISION_API_KEY")
    vision_base_url = get_env_or_secret("VISION_BASE_URL") or vision_config["base_url"]
    vision_model = get_env_or_secret("VISION_MODEL") or vision_config["model"]
    
    local_model = load_local_model_only()

    async def _current_loop_embed(texts):
        return await asyncio.to_thread(lambda: local_model.encode(texts))

    embedding_func = EmbeddingFunc(
        embedding_dim=512, 
        max_token_size=512, 
        func=_current_loop_embed
    )

    # 公式指令
    math_format_instruction = """
    【数学公式规范】
    1. 行内公式用单个$包裹，如 $E=mc^2$。
    2. 块级公式用双$$包裹，必须换行。
    """
    
    # 根据模式构建 Prompt 后缀
    if is_quiz_mode:
        # 测验模式：强制 JSON 输出
        query_suffix = """
        \n\n【任务：生成测验】
        请基于文档内容生成 3 道单项选择题。
        必须严格返回 JSON 数组格式，不要包含 Markdown 标记。格式如下：
        [
            {"question": "题目1", "options": ["A.选项", "B.选项", "C.选项", "D.选项"], "answer": "A", "analysis": "解析"},
            {"question": "题目2", "options": ["A.选项", "B.选项", "C.选项", "D.选项"], "answer": "B", "analysis": "解析"},
            {"question": "题目3", "options": ["A.选项", "B.选项", "C.选项", "D.选项"], "answer": "C", "analysis": "解析"}
        ]
        """
    else:
        # 普通问答模式
        if "初学者" in level:
            query_suffix = f"\n\n【指令：直觉科普模式】用大白话和生活类比解释。\n{math_format_instruction}"
        elif "专家" in level:
            query_suffix = f"\n\n【指令：深度研讨模式】学术、高冷、直击本质。\n{math_format_instruction}"
        else:
            query_suffix = f"\n\n【指令：标准教学模式】定义->公式->物理意义->考点。\n{math_format_instruction}"

    async def safe_deepseek_call(prompt, system_prompt="You are a helpful AI tutor.", history_messages=[], **kwargs):
        kwargs.pop('response_format', None)
        kwargs.pop('keyword_extraction', None)
        
        # 消息清洗
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
        
        # 如果是测验模式，不进行公式处理，直接返回原始 JSON 字符串以便解析
        if is_quiz_mode:
            return raw_text
        return process_math_format(raw_text)

    # 视觉相关函数 (简化保留，不做变动)
    async def vision_func(prompt, **kwargs):
        # ... (此处省略具体视觉逻辑，保持原样即可，为了代码简洁) ...
        # 如果需要完整视觉逻辑请保留您原文件中的 vision_func
        return await safe_deepseek_call(prompt, **kwargs)

    rag = RAGAnything(
        config=RAGAnythingConfig(working_dir="./rag_storage", parser="mineru", parse_method="auto"),
        llm_model_func=safe_deepseek_call,
        vision_model_func=vision_func,
        embedding_func=embedding_func
    )

    if file_path:
        await rag.process_document_complete(file_path=file_path, output_dir="./output", parse_method="auto")

    return await rag.aquery(query + query_suffix, mode="hybrid")

# === 7. 测验逻辑工具函数 ===

def parse_quiz_json(text):
    """解析 LLM 返回的 JSON 题目"""
    try:
        start = text.find('[')
        end = text.rfind(']') + 1
        if start != -1 and end != -1:
            return json.loads(text[start:end])
    except:
        pass
    return None

def calculate_mastery(correct_count):
    """
    判定逻辑：
    - 3题全对 -> 掌握 (100%)
    - 错1题 (对2题) -> 掌握 75%
    - 错2题及以上 -> 未掌握
    """
    if correct_count == 3:
        return "已掌握", 1.0
    elif correct_count == 2:
        return "掌握 75%", 0.75
    else:
        return "未掌握", 0.0

def show_quiz_area(file_path, user_level):
    """显示测验区域"""
    
    # 1. 生成按钮
    if "quiz_data" not in st.session_state:
        st.session_state.quiz_data = None
    
    col1, col2 = st.columns(2)
    with col1:
        btn_text = "📝 生成随堂测验 (3题)" if not st.session_state.quiz_data else "🔄 重新生成测验"
        if st.button(btn_text):
            with st.spinner("🧠 正在基于文档出题..."):
                prompt = "请出3道单项选择题" # 具体 Prompt 在 run_rag 中拼接
                loop = st.session_state.loop
                res = loop.run_until_complete(run_rag(file_path, prompt, user_level, is_quiz_mode=True))
                data = parse_quiz_json(res)
                if data:
                    st.session_state.quiz_data = data
                    st.rerun()
                else:
                    st.error("生成失败，请重试")

    # 2. 渲染题目表单
    if st.session_state.quiz_data:
        st.divider()
        st.markdown("### 🧠 随堂小测验")
        
        with st.form("quiz_form"):
            for idx, q in enumerate(st.session_state.quiz_data):
                st.markdown(f"**Q{idx+1}. {q['question']}**")
                st.radio("选项", q['options'], key=f"q_{idx}", label_visibility="collapsed", index=None)
                st.divider()
            
            submitted = st.form_submit_button("提交答案")
        
        if submitted:
            correct_count = 0
            results = []
            
            # 批改
            for idx, q in enumerate(st.session_state.quiz_data):
                user_val = st.session_state.get(f"q_{idx}")
                user_ans = user_val.split('.')[0].strip() if user_val else ""
                correct_ans = q['answer'].strip()
                
                is_right = (user_ans == correct_ans)
                if is_right: correct_count += 1
                
                results.append({
                    "q": q['question'],
                    "u": user_ans,
                    "c": correct_ans,
                    "ok": is_right,
                    "exp": q['analysis']
                })
            
            # 计算掌握程度
            status_text, score_val = calculate_mastery(correct_count)
            
            # 显示结果
            if status_text == "已掌握":
                st.balloons()
                st.success(f"🎉 3题全对！判定：**{status_text}**")
            elif status_text == "掌握 75%":
                st.info(f"👍 答对 2 题。判定：**{status_text}**")
            else:
                st.error(f"💪 答对 {correct_count} 题。判定：**{status_text}**")
            
            # 详细解析
            with st.expander("查看详细解析", expanded=True):
                for i, r in enumerate(results):
                    icon = "✅" if r['ok'] else "❌"
                    color = "green" if r['ok'] else "red"
                    st.markdown(f"**第{i+1}题** {icon}")
                    st.markdown(f":{color}[你的答案: {r['u']}] | 标准答案: {r['c']}")
                    st.markdown(f"*解析: {r['exp']}*")
                    st.divider()

            # 保存数据
            user_id = get_current_user_id()
            if user_id:
                topic = extract_topic_from_question(str(st.session_state.quiz_data[0]['question']))
                # 假设 update_knowledge_point 支持分数记录，或者您可以在此处调用专门的 save_quiz_record
                # 这里复用 update_knowledge_point，认为 >0.6 即为通过
                is_passed = (score_val >= 0.75)
                update_knowledge_point(user_id, topic, is_correct=is_passed)
                st.toast(f"已记录掌握状态：{status_text}")


# ==================== 主应用逻辑 ====================

def show_chat_page(user_level, file_path):
    """显示问答助手页面"""
    st.title("🎓 新工科 AI 助教系统")
    
    # 获取视觉模型状态用于显示
    _vision_provider = get_env_or_secret("VISION_PROVIDER", "zhipu")
    _model_name = VISION_PROVIDERS.get(_vision_provider, {}).get("model", "")
    vision_status = f"🖼️ {_model_name}" if get_env_or_secret("VISION_API_KEY") else "🖼️ 未配置"
    
    user = get_current_user()
    st.caption(f"👤 {user['username']} | 模式：{user_level} | 引擎：DeepSeek | {vision_status}")

    # 显示聊天记录
    if "messages" not in st.session_state: 
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): 
            st.markdown(msg["content"], unsafe_allow_html=True)

    # === 插入测验区域 ===
    # 只有当上传了文件或有知识库时才允许生成测验
    if file_path or os.path.exists("./rag_storage"):
        show_quiz_area(file_path, user_level)
    else:
        st.info("💡 上传 PDF 教材后即可使用【生成随堂测验】功能")

    # === 普通聊天输入 ===
    if prompt := st.chat_input("请输入你的问题（支持专业公式询问）..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

    # 处理聊天回复
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        last_user_query = st.session_state.messages[-1]["content"]
        
        with st.chat_message("assistant"):
            with st.spinner("🧠 DeepSeek 正在思考..."):
                try:
                    if not file_path and not os.path.exists("./rag_storage"):
                        error_msg = "请先在左侧上传 PDF 教材！"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    else:
                        loop = st.session_state.loop
                        # 普通问答模式
                        response = loop.run_until_complete(run_rag(file_path, last_user_query, user_level, is_quiz_mode=False))
                        
                        try:
                            if isinstance(response, str):
                                final_ans = json.loads(response).get("answer", response)
                            else:
                                final_ans = str(response)
                        except:
                            final_ans = str(response)
                        
                        # 再次清洗以防万一
                        final_ans = process_math_format(final_ans)
                        
                        st.markdown(final_ans, unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": final_ans})
                        
                        # 保存问答记录
                        user_id = get_current_user_id()
                        if user_id:
                            topic = extract_topic_from_question(last_user_query)
                            save_chat_history(user_id, last_user_query, final_ans, topic)
                            # 问答互动默认算作一次正向学习
                            update_knowledge_point(user_id, topic, is_correct=True)
                        
                except Exception as e:
                    st.error(f"发生错误: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})


def main():
    """主函数"""
    if not is_logged_in():
        show_login_page()
        return
    
    user = get_current_user()
    
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=60)
        st.title("⚙️ 学习设置")
        
        page = st.radio("📍 导航", ["💬 问答助手", "📊 学习画像"], index=0)
        st.divider()
        
        user_level = st.radio("我是谁？", 
            ["👶 初学者 (通俗易懂)", "👨‍🎓 本科生 (专业推导)", "👨‍🔬 领域专家 (深度研讨)"], index=1)
        st.divider()
        
        st.header("📂 知识库")
        uploaded_file = st.file_uploader("上传教材 (PDF)", type=["pdf"])
        
        if st.button("🗑️ 清空知识库缓存"):
            import shutil
            if os.path.exists("./rag_storage"): shutil.rmtree("./rag_storage")
            if os.path.exists("./output"): shutil.rmtree("./output")
            # 清除测验缓存
            if "quiz_data" in st.session_state: del st.session_state.quiz_data
            st.success("缓存已清空！请重新上传文件。")
        
        show_user_info_sidebar()
        
        user_id = get_current_user_id()
        if user_id:
            show_mini_profile_card(user_id)
    
    # 处理文件路径
    file_path = None
    if uploaded_file:
        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", uploaded_file.name)
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f: 
                f.write(uploaded_file.getbuffer())
    elif os.path.exists("uploads") and len(os.listdir("uploads")) > 0:
        file_path = os.path.join("uploads", os.listdir("uploads")[0])
    
    # 路由
    if page == "💬 问答助手":
        show_chat_page(user_level, file_path)
    elif page == "📊 学习画像":
        user_id = get_current_user_id()
        if user_id:
            show_profile_page(user_id)
        else:
            st.error("无法获取用户信息")

if __name__ == "__main__":
    main()