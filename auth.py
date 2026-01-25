"""
用户认证模块 - 处理注册、登录、密码加密
"""

import bcrypt
import streamlit as st
from typing import Optional, Dict
from database import (
    init_database, 
    create_user, 
    get_user_by_username, 
    update_user_level,
    start_study_session,
    end_study_session
)


def hash_password(password: str) -> str:
    """对密码进行哈希加密"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(password: str, password_hash: str) -> bool:
    """验证密码是否正确"""
    return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))


def register_user(username: str, password: str, level: str = "本科生") -> tuple[bool, str]:
    """
    注册新用户
    返回: (成功与否, 消息)
    """
    # 验证输入
    if not username or len(username) < 2:
        return False, "用户名至少需要2个字符"
    
    if not password or len(password) < 4:
        return False, "密码至少需要4个字符"
    
    # 检查用户名是否已存在
    existing_user = get_user_by_username(username)
    if existing_user:
        return False, "用户名已被注册"
    
    # 创建用户
    password_hash = hash_password(password)
    user_id = create_user(username, password_hash, level)
    
    if user_id:
        return True, "注册成功！请登录"
    else:
        return False, "注册失败，请重试"


def login_user(username: str, password: str) -> tuple[bool, str, Optional[Dict]]:
    """
    用户登录
    返回: (成功与否, 消息, 用户信息)
    """
    if not username or not password:
        return False, "请输入用户名和密码", None
    
    user = get_user_by_username(username)
    if not user:
        return False, "用户名不存在", None
    
    if not verify_password(password, user['password_hash']):
        return False, "密码错误", None
    
    return True, "登录成功！", user


def logout_user():
    """用户登出"""
    # 结束学习会话
    if "study_session_id" in st.session_state:
        end_study_session(st.session_state.study_session_id)
        del st.session_state.study_session_id
    
    # 清除用户信息
    if "user" in st.session_state:
        del st.session_state.user
    if "user_id" in st.session_state:
        del st.session_state.user_id
    
    # 清除聊天历史
    if "messages" in st.session_state:
        del st.session_state.messages


def is_logged_in() -> bool:
    """检查用户是否已登录"""
    return "user" in st.session_state and st.session_state.user is not None


def get_current_user() -> Optional[Dict]:
    """获取当前登录用户信息"""
    if is_logged_in():
        return st.session_state.user
    return None


def get_current_user_id() -> Optional[int]:
    """获取当前登录用户ID"""
    if is_logged_in():
        return st.session_state.user.get('id')
    return None


def show_login_page():
    """显示登录/注册页面"""
    st.markdown("""
    <style>
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎓 新工科 AI 助教系统")
    st.subheader("欢迎使用个性化学习平台")
    
    # 初始化数据库
    init_database()
    
    # 选择登录或注册
    tab1, tab2 = st.tabs(["🔐 登录", "📝 注册"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("用户名", key="login_username")
            password = st.text_input("密码", type="password", key="login_password")
            submit = st.form_submit_button("登录", use_container_width=True)
            
            if submit:
                success, message, user = login_user(username, password)
                if success:
                    st.session_state.user = user
                    st.session_state.user_id = user['id']
                    # 开始学习会话
                    session_id = start_study_session(user['id'])
                    st.session_state.study_session_id = session_id
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
    
    with tab2:
        with st.form("register_form"):
            new_username = st.text_input("用户名", key="reg_username")
            new_password = st.text_input("密码", type="password", key="reg_password")
            confirm_password = st.text_input("确认密码", type="password", key="reg_confirm")
            level = st.selectbox("学习水平", 
                ["👶 初学者 (通俗易懂)", "👨‍🎓 本科生 (专业推导)", "👨‍🔬 领域专家 (深度研讨)"],
                index=1,
                key="reg_level"
            )
            submit = st.form_submit_button("注册", use_container_width=True)
            
            if submit:
                if new_password != confirm_password:
                    st.error("两次输入的密码不一致")
                else:
                    success, message = register_user(new_username, new_password, level)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
    
    # 页脚信息
    st.divider()
    st.caption("💡 提示：首次使用请先注册账号，系统将为您记录学习进度和个性化分析")


def show_user_info_sidebar():
    """在侧边栏显示用户信息"""
    if is_logged_in():
        user = get_current_user()
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"👤 **当前用户**: {user['username']}")
        st.sidebar.markdown(f"📊 **学习等级**: {user['level']}")
        
        if st.sidebar.button("🚪 退出登录", use_container_width=True):
            logout_user()
            st.rerun()
