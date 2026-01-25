"""
学生画像仪表盘页面 - 展示学习概况、知识点雷达图、学习趋势等
"""

import streamlit as st
from analytics import LearningAnalytics
from database import get_chat_history, get_quiz_results


def show_profile_page(user_id: int):
    """显示学生画像仪表盘"""
    
    st.title("📊 我的学习画像")
    st.caption("基于您的学习数据生成的个性化分析报告")
    
    # 初始化分析引擎
    analytics = LearningAnalytics(user_id)
    
    # 获取概况统计
    stats = analytics.get_overview_stats()
    
    # === 1. 学习概况卡片 ===
    st.subheader("📈 学习概况")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📝 累计提问",
            value=f"{stats['total_questions']} 个",
            help="您向AI助教提出的问题总数"
        )
    
    with col2:
        hours = stats['total_study_hours']
        if hours >= 1:
            time_display = f"{hours:.1f} 小时"
        else:
            time_display = f"{stats['total_study_minutes']:.0f} 分钟"
        st.metric(
            label="⏱️ 学习时长",
            value=time_display,
            help="您在平台上的累计学习时间"
        )
    
    with col3:
        accuracy = stats['quiz_accuracy']
        delta_color = "normal" if accuracy >= 60 else "inverse"
        st.metric(
            label="🎯 测验正确率",
            value=f"{accuracy}%",
            delta="优秀" if accuracy >= 80 else ("良好" if accuracy >= 60 else "需加强"),
            delta_color="off"
        )
    
    with col4:
        mastery = stats['avg_mastery']
        st.metric(
            label="💡 平均掌握度",
            value=f"{mastery}%",
            help=f"已学习 {stats['topics_learned']} 个知识点"
        )
    
    st.divider()
    
    # === 2. 图表区域 ===
    col_left, col_right = st.columns(2)
    
    with col_left:
        # 知识点雷达图
        st.subheader("🎯 知识点掌握雷达图")
        radar_chart = analytics.create_knowledge_radar_chart()
        if radar_chart:
            st.plotly_chart(radar_chart, use_container_width=True)
        else:
            st.info("📚 开始学习后，这里将显示您的知识点掌握情况")
    
    with col_right:
        # 主题分布饼图
        st.subheader("📊 学习主题分布")
        pie_chart = analytics.create_topic_pie_chart()
        if pie_chart:
            st.plotly_chart(pie_chart, use_container_width=True)
        else:
            st.info("📚 开始提问后，这里将显示您的学习主题分布")
    
    # === 3. 学习趋势 ===
    st.subheader("📅 学习趋势")
    
    # 选择时间范围
    time_range = st.radio(
        "选择时间范围",
        options=[7, 14, 30],
        format_func=lambda x: f"最近 {x} 天",
        horizontal=True,
        index=1
    )
    
    trend_chart = analytics.create_study_trend_chart(time_range)
    st.plotly_chart(trend_chart, use_container_width=True)
    
    st.divider()
    
    # === 4. 知识点掌握度排行 ===
    col_mastery, col_suggestions = st.columns([3, 2])
    
    with col_mastery:
        st.subheader("📉 知识点掌握度排行")
        mastery_chart = analytics.create_mastery_bar_chart()
        if mastery_chart:
            st.plotly_chart(mastery_chart, use_container_width=True)
        else:
            st.info("📚 完成测验后，这里将显示各知识点的掌握程度")
    
    with col_suggestions:
        st.subheader("💡 学习建议")
        suggestions = analytics.get_learning_suggestions()
        
        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")
        
        # 薄弱环节提示
        weak_points = analytics.get_weak_topics()
        if weak_points:
            st.warning("⚠️ **需要重点关注的知识点：**")
            for wp in weak_points[:5]:
                score = round(wp['mastery_score'] * 100, 1)
                st.markdown(f"- {wp['topic']}（掌握度：{score}%）")
    
    st.divider()
    
    # === 5. 最近学习记录 ===
    with st.expander("📜 最近问答记录", expanded=False):
        history = get_chat_history(user_id, limit=10)
        if history:
            for i, record in enumerate(history):
                with st.container():
                    st.markdown(f"**Q{i+1}:** {record['question'][:100]}...")
                    st.caption(f"🏷️ 主题: {record['topic']} | 🕐 {record['timestamp']}")
                    if i < len(history) - 1:
                        st.markdown("---")
        else:
            st.info("暂无问答记录")
    
    with st.expander("📝 最近测验记录", expanded=False):
        quiz_results = get_quiz_results(user_id, limit=10)
        if quiz_results:
            for i, record in enumerate(quiz_results):
                status = "✅" if record['is_correct'] else "❌"
                st.markdown(f"{status} **{record['question'][:80]}...**")
                st.caption(f"🏷️ 主题: {record['topic']} | 🕐 {record['timestamp']}")
                if i < len(quiz_results) - 1:
                    st.markdown("---")
        else:
            st.info("暂无测验记录")


def show_mini_profile_card(user_id: int):
    """显示迷你学习卡片（用于侧边栏）"""
    analytics = LearningAnalytics(user_id)
    stats = analytics.get_overview_stats()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 今日学习")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("提问", f"{stats['total_questions']}")
    with col2:
        st.metric("掌握度", f"{stats['avg_mastery']}%")
    
    # 薄弱提示
    weak_points = analytics.get_weak_topics()
    if weak_points:
        st.sidebar.warning(f"⚠️ 有 {len(weak_points)} 个薄弱知识点需要关注")
