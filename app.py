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

# === 视觉模型配置 ===
# 支持的视觉模型提供商配置
VISION_PROVIDERS = {
    "zhipu": {  # 智谱 AI GLM-4V (推荐)
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "model": "glm-4v",
        "env_key": "ZHIPU_API_KEY"
    },
    "qwen": {  # 阿里通义千问
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-vl-max",
        "env_key": "QWEN_API_KEY"
    },
    "siliconflow": {  # 硅基流动
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

# === 4. 辅助函数：清洗数学公式格式 ===
def process_math_format(text):
    """
    处理 LLM 返回的文本中的数学公式，确保能被 Streamlit 正确渲染
    Streamlit 支持 $...$ (行内) 和 $$...$$ (块级) 格式的 LaTeX
    """
    if not isinstance(text, str): return str(text)
    
    # 1. 将 \(...\) 转换为 $...$
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
    
    # 2. 将 \[...\] 转换为 $$...$$
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    
    # 3. 处理反引号中的公式内容
    def remove_code_ticks(match):
        content = match.group(1)
        if '\\' in content or '^' in content or '_' in content:
            return f"${content.strip('$')}$"
        return match.group(0)
    text = re.sub(r'`([^`]+)`', remove_code_ticks, text)
    
    # 4. 核心逻辑：按行处理，找到公式段落并包裹
    def wrap_latex_in_line(line):
        """处理单行文本，包裹其中的 LaTeX 公式"""
        # 如果行中没有反斜杠或已经有 $，跳过
        if '\\' not in line:
            return line
        if line.strip().startswith('$') and line.strip().endswith('$'):
            return line
        
        # 中文字符范围
        def is_chinese(char):
            return '\u4e00' <= char <= '\u9fff' or char in '，。、；：""''（）【】！？'
        
        result = []
        i = 0
        n = len(line)
        
        while i < n:
            char = line[i]
            
            # 如果已经在 $ 内，直接跳过直到 $
            if char == '$':
                j = i + 1
                while j < n and line[j] != '$':
                    j += 1
                result.append(line[i:j+1] if j < n else line[i:])
                i = j + 1
                continue
            
            # 检测 LaTeX 公式开始
            if char == '\\' and i + 1 < n and line[i + 1].isalpha():
                latex_start = i
                
                # 使用平衡括号法找到公式结束位置
                brace_depth = 0
                j = i
                last_valid_end = i
                
                while j < n:
                    c = line[j]
                    
                    # 遇到中文，公式结束
                    if is_chinese(c):
                        break
                    
                    if c == '{':
                        brace_depth += 1
                        j += 1
                        last_valid_end = j
                    elif c == '}':
                        brace_depth -= 1
                        j += 1
                        last_valid_end = j
                        # 如果括号平衡了，检查后面是否还有公式内容
                        if brace_depth == 0:
                            # 跳过空格
                            k = j
                            while k < n and line[k] == ' ':
                                k += 1
                            # 检查后面是否还有公式相关字符
                            if k < n and line[k] in '\\=+-^_{}':
                                j = k
                                continue
                            elif k < n and line[k] == '{':
                                # 可能是 \frac{}{} 的第二个参数
                                j = k
                                continue
                    elif c == '\\' and j + 1 < n and line[j + 1].isalpha():
                        # 另一个 LaTeX 命令
                        j += 1
                        while j < n and (line[j].isalnum() or line[j] == '*'):
                            j += 1
                        last_valid_end = j
                    elif c in '^_':
                        j += 1
                        if j < n and line[j] == '{':
                            brace_depth += 1
                            j += 1
                        elif j < n and (line[j].isalnum() or line[j] == '\\'):
                            j += 1
                        last_valid_end = j
                    elif c.isalnum() or c in '.,+-=|<>()[]':
                        j += 1
                        last_valid_end = j
                    elif c == ' ':
                        # 空格：检查后面是否还有公式内容
                        k = j + 1
                        while k < n and line[k] == ' ':
                            k += 1
                        if k < n and (line[k] in '\\=+-^_{}' or line[k].isalnum()):
                            j = k
                        else:
                            break
                    else:
                        break
                
                # 确保括号平衡
                if brace_depth != 0:
                    j = last_valid_end
                
                latex_expr = line[latex_start:j].strip()
                
                # 移除尾部的标点
                while latex_expr and latex_expr[-1] in '，。、；：':
                    latex_expr = latex_expr[:-1]
                
                if latex_expr:
                    result.append(f'${latex_expr}$')
                
                i = j
            else:
                result.append(char)
                i += 1
        
        return ''.join(result)
    
    # 按行处理
    lines = text.split('\n')
    processed_lines = [wrap_latex_in_line(line) for line in lines]
    text = '\n'.join(processed_lines)
    
    # 5. 修复可能产生的问题
    # 修复连续的 $$ 
    text = re.sub(r'\$\$+', '$', text)  # 多个 $ 变成 1 个
    text = re.sub(r'\$\s*\$', '', text)  # 移除空的 $$
    
    # 6. 修复被错误拆分的公式（如 $a^2$ = $b^2$ 应该是 $a^2 = b^2$）
    def merge_adjacent_formulas(text):
        # 匹配 $...$空格=空格$...$ 这样的模式并合并
        pattern = r'\$([^$]+)\$(\s*[=<>+\-]\s*)\$([^$]+)\$'
        while re.search(pattern, text):
            text = re.sub(pattern, r'$\1\2\3$', text)
        return text
    
    text = merge_adjacent_formulas(text)
    
    # 7. 确保 $...$ 之间没有换行（否则 Streamlit 不会渲染）
    def fix_multiline_inline_math(match):
        content = match.group(1)
        if '\n' in content:
            return f'$${content}$$'
        return match.group(0)
    
    text = re.sub(r'\$([^$]+)\$', fix_multiline_inline_math, text)
    
    return text

# === 5. 模型加载 ===
@st.cache_resource
def load_local_model_only():
    print("正在加载本地 BGE-Small 中文模型...")
    return SentenceTransformer('BAAI/bge-small-zh-v1.5')

# === 6. 核心 RAG 业务逻辑 ===
async def run_rag(file_path, query, level):
    # DeepSeek 配置 (文本处理)
    api_key = os.getenv("LLM_BINDING_API_KEY")
    base_url = os.getenv("LLM_BINDING_HOST")
    
    # 视觉模型配置 (图像处理)
    vision_provider = os.getenv("VISION_PROVIDER", "zhipu")  # 默认使用智谱
    vision_config = VISION_PROVIDERS.get(vision_provider, VISION_PROVIDERS["zhipu"])
    vision_api_key = os.getenv(vision_config["env_key"]) or os.getenv("VISION_API_KEY")
    vision_base_url = os.getenv("VISION_BASE_URL") or vision_config["base_url"]
    vision_model = os.getenv("VISION_MODEL") or vision_config["model"]
    
    local_model = load_local_model_only()

    async def _current_loop_embed(texts):
        return await asyncio.to_thread(lambda: local_model.encode(texts))

    embedding_func = EmbeddingFunc(
        embedding_dim=512, 
        max_token_size=512, 
        func=_current_loop_embed
    )

    # 普适性增强 Prompt
    # 公式格式要求（所有模式通用）
    math_format_instruction = """
【数学公式格式要求】
- 行内公式必须用单个美元符号包裹，如：$a^2 + b^2 = c^2$
- 块级公式必须用双美元符号包裹，如：$$\\frac{\\partial u}{\\partial t} = 0$$
- 禁止使用 \\( \\) 或 \\[ \\] 格式
- 所有希腊字母如 $\\varphi$, $\\alpha$, $\\partial$ 等必须用 $ 包裹
"""
    
    query_suffix = ""
    if "初学者" in level:
        query_suffix = f"""\n\n【指令：直觉科普模式】
        1. 🚫 严禁使用晦涩专业术语，必须用大白话。
        2. ✅ 核心：使用生活中的类比（如把电路比作水管）。
        3. 语气：幽默风趣的科普博主。
        {math_format_instruction}
        """
    elif "专家" in level:
        query_suffix = f"""\n\n【指令：深度研讨模式】
        1. ⚠️ 跳过基础定义，假设用户是同行。
        2. ✅ 核心：切入问题本质、底层机制、局限性。
        3. 语气：极度简练、学术、高冷。
        {math_format_instruction}
        """
    else:
        query_suffix = f"""\n\n【指令：标准教学模式】
        1. 目标：帮助通过期末考试。
        2. ✅ 结构：定义 -> 公式 -> 物理意义 -> 考点。
        3. 语气：耐心的大学助教。
        {math_format_instruction}
        """

    # DeepSeek 文本模型调用
    async def safe_deepseek_call(prompt, system_prompt="You are a helpful AI tutor.", history_messages=[], **kwargs):
        # DeepSeek 不支持 response_format 和 keyword_extraction 功能
        kwargs.pop('response_format', None)
        kwargs.pop('keyword_extraction', None)
        
        # 清洗图片消息 (DeepSeek 不支持图片)
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

    # 视觉模型调用 (支持图像处理)
    # 不同模型的图片数量限制：
    #   - 智谱 GLM-4V: 1 张
    #   - 阿里 Qwen-VL-Max: 约 10 张
    #   - 硅基流动 Qwen2-VL: 约 10 张
    vision_provider = os.getenv("VISION_PROVIDER", "zhipu").lower()
    if vision_provider == "qwen":
        MAX_IMAGES_PER_REQUEST = 10  # 阿里 Qwen-VL-Max 支持多图
    elif vision_provider == "siliconflow":
        MAX_IMAGES_PER_REQUEST = 10  # 硅基流动也支持多图
    else:
        MAX_IMAGES_PER_REQUEST = 1   # 智谱 GLM-4V 限制为 1 张
    
    # 辅助函数：从 messages 提取纯文本和图片
    def extract_content_from_messages(messages):
        """从 messages 中分离文本和图片"""
        text_parts = []
        images = []
        system_content = None
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content")
            
            if role == "system":
                system_content = content if isinstance(content, str) else ""
                continue
                
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        images.append(item)
            elif isinstance(content, str):
                text_parts.append(content)
                
        return system_content, "\n".join(text_parts), images
    
    # 辅助函数：构建单批次的 VLM 消息
    def build_batch_messages(system_prompt, text_content, batch_images, batch_num=None, total_batches=None):
        """构建单批次的 VLM 消息格式"""
        content_parts = []
        
        # 添加文本内容
        if batch_num and total_batches and total_batches > 1:
            batch_info = f"\n\n[这是第 {batch_num}/{total_batches} 批图片分析]"
            content_parts.append({"type": "text", "text": text_content + batch_info})
        else:
            content_parts.append({"type": "text", "text": text_content})
        
        # 添加图片
        for img in batch_images:
            content_parts.append(img)
        
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": content_parts})
        
        return msgs
    
    async def vision_func(prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs):
        # VLM 不支持 response_format 和 keyword_extraction 功能
        kwargs.pop('response_format', None)
        kwargs.pop('keyword_extraction', None)
        
        # 如果没有配置视觉 API Key，回退到 DeepSeek (仅文本)
        if not vision_api_key:
            print("⚠️ 未配置视觉模型 API Key，回退到纯文本模式")
            return await safe_deepseek_call(prompt, system_prompt, history_messages, **kwargs)
        
        # 如果提供了 messages 格式 (多模态 VLM 增强查询)
        if messages:
            # 提取系统提示、文本内容和所有图片
            sys_prompt, text_content, all_images = extract_content_from_messages(messages)
            
            if not all_images:
                # 没有图片，回退到纯文本模式
                return await safe_deepseek_call(text_content or prompt, sys_prompt or system_prompt, history_messages, **kwargs)
            
            total_images = len(all_images)
            print(f"📷 检测到 {total_images} 张图片，每批最多 {MAX_IMAGES_PER_REQUEST} 张")
            print(f"🔧 使用视觉模型: {vision_model} @ {vision_base_url}")
            
            # 如果图片数量在限制内，直接处理
            if total_images <= MAX_IMAGES_PER_REQUEST:
                batch_messages = build_batch_messages(sys_prompt or system_prompt, text_content, all_images)
                try:
                    print(f"🚀 正在调用 VLM: {vision_model}...")
                    response = await openai_complete_if_cache(
                        vision_model, "",
                        system_prompt=None, history_messages=[],
                        messages=batch_messages,
                        api_key=vision_api_key, base_url=vision_base_url, **kwargs
                    )
                    print(f"✅ VLM 调用成功！响应长度: {len(str(response))} 字符")
                    raw_text = response.replace("```json", "").replace("```", "").strip() if isinstance(response, str) else str(response)
                    return process_math_format(raw_text)
                except Exception as e:
                    print(f"❌ VLM 调用失败: {e}")
                    print(f"⚠️ 回退到 DeepSeek 纯文本模式")
                    return await safe_deepseek_call(text_content or prompt, sys_prompt or system_prompt, history_messages, **kwargs)
            
            # === 图片分批处理 ===
            # 将图片分成多个批次
            batches = []
            for i in range(0, total_images, MAX_IMAGES_PER_REQUEST):
                batch = all_images[i:i + MAX_IMAGES_PER_REQUEST]
                batches.append(batch)
            
            total_batches = len(batches)
            print(f"📦 将 {total_images} 张图片分成 {total_batches} 批处理")
            
            # 处理每个批次
            batch_results = []
            for batch_idx, batch_images in enumerate(batches, 1):
                print(f"🔄 正在处理第 {batch_idx}/{total_batches} 批 ({len(batch_images)} 张图片)...")
                
                batch_messages = build_batch_messages(
                    sys_prompt or system_prompt, 
                    text_content, 
                    batch_images,
                    batch_num=batch_idx,
                    total_batches=total_batches
                )
                
                try:
                    response = await openai_complete_if_cache(
                        vision_model, "",
                        system_prompt=None, history_messages=[],
                        messages=batch_messages,
                        api_key=vision_api_key, base_url=vision_base_url, **kwargs
                    )
                    result = response.replace("```json", "").replace("```", "").strip() if isinstance(response, str) else str(response)
                    batch_results.append(f"【第 {batch_idx} 批图片分析】\n{result}")
                    print(f"✅ 第 {batch_idx} 批处理完成")
                except Exception as e:
                    print(f"⚠️ 第 {batch_idx} 批 VLM 调用失败: {e}")
                    batch_results.append(f"【第 {batch_idx} 批图片分析失败】")
            
            # 如果只有一个批次成功，直接返回
            if len(batch_results) == 1:
                return process_math_format(batch_results[0])
            
            # 使用 DeepSeek 综合所有批次的结果
            print("🧠 正在用 DeepSeek 综合所有批次的分析结果...")
            combined_prompt = f"""请综合以下多批次的图片分析结果，给出完整、连贯的回答：

{chr(10).join(batch_results)}

---
原始问题上下文：
{text_content[:2000]}...

请基于以上所有批次的分析，给出统一、完整的回答。"""
            
            final_response = await safe_deepseek_call(
                combined_prompt,
                system_prompt="你是一个专业的分析助手，请综合多个批次的图片分析结果，给出完整连贯的回答。",
                history_messages=[]
            )
            return process_math_format(final_response)
                
        # 如果提供了单张图片
        elif image_data:
            built_messages = []
            if system_prompt:
                built_messages.append({"role": "system", "content": system_prompt})
            built_messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                ]
            })
            try:
                response = await openai_complete_if_cache(
                    vision_model, "",
                    system_prompt=None, history_messages=[],
                    messages=built_messages,
                    api_key=vision_api_key, base_url=vision_base_url, **kwargs
                )
            except Exception as e:
                print(f"⚠️ VLM 调用失败: {e}，回退到纯文本模式")
                return await safe_deepseek_call(prompt, system_prompt, history_messages, **kwargs)
        # 纯文本，使用 DeepSeek
        else:
            return await safe_deepseek_call(prompt, system_prompt, history_messages, **kwargs)
        
        raw_text = response.replace("```json", "").replace("```", "").strip() if isinstance(response, str) else str(response)
        return process_math_format(raw_text)

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
# 获取视觉模型信息显示
_vision_provider = os.getenv("VISION_PROVIDER", "zhipu")
_vision_model_name = VISION_PROVIDERS.get(_vision_provider, {}).get("model", "未配置")
if os.getenv("VISION_API_KEY") or os.getenv(VISION_PROVIDERS.get(_vision_provider, {}).get("env_key", "")):
    vision_status = f"🖼️ {_vision_model_name}"
else:
    vision_status = "🖼️ 未配置"
st.caption(f"当前模式：{user_level} | 文本引擎：DeepSeek | 视觉引擎：{vision_status} | Embedding：BGE-Small")

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