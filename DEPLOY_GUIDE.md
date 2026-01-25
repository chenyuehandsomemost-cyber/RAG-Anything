# Streamlit Cloud 部署指南

本文档说明如何将「新工科 AI 助教系统」部署到 Streamlit Cloud，使其可以通过公网 URL 访问。

## 一、准备工作

### 1. 确保代码已推送到 GitHub

```bash
git add .
git commit -m "添加学生画像系统和部署配置"
git push origin main
```

### 2. 检查必要文件

确保仓库中包含以下文件：
- `app.py` - 主应用入口
- `requirements.txt` - Python 依赖
- `.streamlit/config.toml` - Streamlit 配置
- `.streamlit/secrets.toml.example` - Secrets 模板（参考用）

## 二、Streamlit Cloud 部署步骤

### 步骤 1：登录 Streamlit Cloud

1. 访问 [share.streamlit.io](https://share.streamlit.io)
2. 使用 GitHub 账号登录

### 步骤 2：创建新应用

1. 点击 **New app** 按钮
2. 选择你的 GitHub 仓库
3. 填写以下信息：
   - **Repository**: `你的用户名/RAG-Anything`
   - **Branch**: `main`
   - **Main file path**: `app.py`

### 步骤 3：配置 Secrets

**这是最关键的一步！**

1. 点击 **Advanced settings**
2. 在 **Secrets** 文本框中输入以下内容：

```toml
[api_keys]
# DeepSeek 文本模型
LLM_BINDING_API_KEY = "sk-你的DeepSeek-API-Key"
LLM_BINDING_HOST = "https://api.deepseek.com/v1"

# 视觉模型（推荐使用阿里通义）
VISION_PROVIDER = "qwen"
QWEN_API_KEY = "sk-你的通义千问-API-Key"
```

> ⚠️ **重要提示**：
> - 请替换为你的真实 API Key
> - 这些 Secrets 不会被公开，只在服务器端使用

### 步骤 4：点击 Deploy

等待部署完成（通常需要 3-5 分钟）。

## 三、部署后访问

部署成功后，你将获得一个类似这样的 URL：

```
https://你的应用名.streamlit.app
```

将此 URL 分享给其他人即可访问。

## 四、常见问题

### Q1: 部署失败，提示依赖安装错误

**解决方案**：
1. 检查 `requirements.txt` 中的依赖是否正确
2. 部分包（如 `mineru`）可能需要特定版本，可以尝试固定版本号

### Q2: 应用启动后显示 API Key 错误

**解决方案**：
1. 进入 App Settings → Secrets
2. 检查 Secrets 配置是否正确
3. 确保 API Key 有效且未过期

### Q3: 数据在重启后丢失

**原因**：Streamlit Cloud 免费版的文件系统是临时的。

**解决方案**（可选升级）：
1. 使用云数据库替代 SQLite，如：
   - [Supabase](https://supabase.com) (免费 PostgreSQL)
   - [TiDB Cloud](https://tidbcloud.com) (免费分布式数据库)
   - [MongoDB Atlas](https://www.mongodb.com/atlas) (免费 MongoDB)

2. 修改 `database.py` 连接云数据库

### Q4: 应用加载很慢

**原因**：首次加载需要下载模型文件。

**解决方案**：
- 模型下载后会被缓存，后续访问会变快
- 可以考虑使用更小的嵌入模型

## 五、资源限制（免费版）

| 资源 | 限制 |
|------|------|
| 内存 | 1 GB |
| CPU | 共享 |
| 存储 | 临时（重启后重置） |
| 并发用户 | 有限 |
| 休眠 | 无活动 7 天后休眠 |

如需更高配置，可考虑：
- Streamlit Cloud 付费版
- 自行部署到云服务器（阿里云/腾讯云）

## 六、本地测试

在部署前，建议先在本地测试：

```bash
# 安装依赖
pip install -r requirements.txt

# 创建本地 secrets 文件
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# 编辑 secrets.toml，填入真实的 API Key

# 启动应用
streamlit run app.py
```

访问 http://localhost:8501 进行测试。

---

## 附录：完整的 Secrets 配置示例

```toml
[api_keys]
# === 文本模型 (必填) ===
LLM_BINDING_API_KEY = "sk-xxx"
LLM_BINDING_HOST = "https://api.deepseek.com/v1"

# === 视觉模型 (选填，但推荐配置) ===
VISION_PROVIDER = "qwen"  # 可选: zhipu, qwen, siliconflow

# 阿里通义千问 (推荐，支持多图)
QWEN_API_KEY = "sk-xxx"

# 或 智谱 GLM-4V
# ZHIPU_API_KEY = "xxx"

# 或 硅基流动
# SILICONFLOW_API_KEY = "xxx"
```

---

如有问题，欢迎在 GitHub Issues 中反馈！
