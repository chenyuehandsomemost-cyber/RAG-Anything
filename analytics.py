"""
学习分析引擎 - 知识点掌握度、薄弱环节识别、学习曲线生成
"""

import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd

from database import (
    get_user_statistics,
    get_knowledge_points,
    get_weak_points,
    get_daily_study_time,
    get_chat_history,
    get_quiz_results,
    get_quiz_accuracy_by_topic,
    get_total_study_time,
    get_total_questions,
    get_quiz_accuracy
)


class LearningAnalytics:
    """学习分析引擎"""
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self._cache = {}
    
    def get_overview_stats(self) -> Dict[str, Any]:
        """获取学习概况统计"""
        total_questions = get_total_questions(self.user_id)
        total_minutes = get_total_study_time(self.user_id)
        accuracy = get_quiz_accuracy(self.user_id)
        knowledge_points = get_knowledge_points(self.user_id)
        
        # 计算平均掌握度
        avg_mastery = 0.0
        if knowledge_points:
            avg_mastery = sum(kp['mastery_score'] for kp in knowledge_points) / len(knowledge_points)
        
        return {
            "total_questions": total_questions,
            "total_study_hours": round(total_minutes / 60, 1),
            "total_study_minutes": round(total_minutes, 0),
            "quiz_accuracy": round(accuracy * 100, 1),
            "avg_mastery": round(avg_mastery * 100, 1),
            "topics_learned": len(knowledge_points)
        }
    
    def get_knowledge_radar_data(self) -> Dict[str, Any]:
        """生成知识点雷达图数据"""
        knowledge_points = get_knowledge_points(self.user_id)
        
        if not knowledge_points:
            return {"topics": [], "scores": [], "has_data": False}
        
        # 取前8个主要知识点（按问题数量排序）
        sorted_kps = sorted(knowledge_points, key=lambda x: x['question_count'], reverse=True)[:8]
        
        topics = [kp['topic'][:10] + "..." if len(kp['topic']) > 10 else kp['topic'] for kp in sorted_kps]
        scores = [round(kp['mastery_score'] * 100, 1) for kp in sorted_kps]
        
        return {
            "topics": topics,
            "scores": scores,
            "has_data": True
        }
    
    def create_knowledge_radar_chart(self) -> Optional[go.Figure]:
        """创建知识点雷达图"""
        data = self.get_knowledge_radar_data()
        
        if not data["has_data"]:
            return None
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=data["scores"],
            theta=data["topics"],
            fill='toself',
            name='掌握度',
            line_color='rgb(31, 119, 180)',
            fillcolor='rgba(31, 119, 180, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    ticksuffix='%'
                )
            ),
            showlegend=False,
            title="知识点掌握度雷达图",
            height=400
        )
        
        return fig
    
    def get_study_trend_data(self, days: int = 14) -> pd.DataFrame:
        """获取学习趋势数据"""
        daily_data = get_daily_study_time(self.user_id, days)
        
        # 创建完整的日期范围
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days-1)
        date_range = pd.date_range(start=start_date, end=end_date)
        
        # 转换为 DataFrame
        df = pd.DataFrame(date_range, columns=['date'])
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        df['minutes'] = 0.0
        
        # 填充实际数据
        for record in daily_data:
            mask = df['date'] == record['date']
            if mask.any():
                df.loc[mask, 'minutes'] = record['minutes']
        
        return df
    
    def create_study_trend_chart(self, days: int = 14) -> go.Figure:
        """创建学习趋势折线图"""
        df = self.get_study_trend_data(days)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['minutes'],
            mode='lines+markers',
            name='学习时长',
            line=dict(color='rgb(55, 83, 109)', width=2),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(55, 83, 109, 0.1)'
        ))
        
        fig.update_layout(
            title=f"最近{days}天学习时长趋势",
            xaxis_title="日期",
            yaxis_title="学习时长（分钟）",
            height=350,
            xaxis=dict(
                tickangle=45,
                tickmode='auto',
                nticks=7
            )
        )
        
        return fig
    
    def get_topic_distribution(self) -> Dict[str, int]:
        """获取主题分布"""
        from database import get_chat_count_by_topic
        return get_chat_count_by_topic(self.user_id)
    
    def create_topic_pie_chart(self) -> Optional[go.Figure]:
        """创建主题分布饼图"""
        distribution = self.get_topic_distribution()
        
        if not distribution:
            return None
        
        # 取前6个主题，其余归为"其他"
        sorted_topics = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_topics) > 6:
            main_topics = dict(sorted_topics[:5])
            other_count = sum(count for _, count in sorted_topics[5:])
            main_topics["其他"] = other_count
        else:
            main_topics = dict(sorted_topics)
        
        fig = go.Figure(data=[go.Pie(
            labels=list(main_topics.keys()),
            values=list(main_topics.values()),
            hole=0.4,
            textinfo='label+percent',
            marker=dict(colors=px.colors.qualitative.Set3)
        )])
        
        fig.update_layout(
            title="学习主题分布",
            height=350,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2)
        )
        
        return fig
    
    def get_weak_topics(self, threshold: float = 0.6) -> List[Dict]:
        """获取薄弱知识点"""
        return get_weak_points(self.user_id, threshold)
    
    def create_mastery_bar_chart(self) -> Optional[go.Figure]:
        """创建知识点掌握度柱状图"""
        knowledge_points = get_knowledge_points(self.user_id)
        
        if not knowledge_points:
            return None
        
        # 按掌握度排序
        sorted_kps = sorted(knowledge_points, key=lambda x: x['mastery_score'])[:10]
        
        topics = [kp['topic'][:15] + "..." if len(kp['topic']) > 15 else kp['topic'] for kp in sorted_kps]
        scores = [round(kp['mastery_score'] * 100, 1) for kp in sorted_kps]
        
        # 根据掌握度设置颜色
        colors = ['#ff6b6b' if s < 60 else '#ffd93d' if s < 80 else '#6bcb77' for s in scores]
        
        fig = go.Figure(data=[go.Bar(
            x=scores,
            y=topics,
            orientation='h',
            marker_color=colors,
            text=[f"{s}%" for s in scores],
            textposition='outside'
        )])
        
        fig.update_layout(
            title="知识点掌握度排行（低到高）",
            xaxis_title="掌握度 (%)",
            yaxis_title="",
            height=400,
            xaxis=dict(range=[0, 110]),
            yaxis=dict(autorange="reversed")
        )
        
        return fig
    
    def get_learning_suggestions(self) -> List[str]:
        """基于分析生成学习建议"""
        suggestions = []
        
        stats = self.get_overview_stats()
        weak_points = self.get_weak_topics()
        
        # 基于学习时长建议
        if stats['total_study_hours'] < 1:
            suggestions.append("📚 建议增加学习时间，每天至少学习30分钟可以有效提升知识掌握")
        
        # 基于正确率建议
        if stats['quiz_accuracy'] < 60:
            suggestions.append("🎯 测验正确率较低，建议复习基础概念后再做练习")
        elif stats['quiz_accuracy'] < 80:
            suggestions.append("📈 正确率还有提升空间，建议针对错题进行专项复习")
        
        # 基于薄弱环节建议
        if weak_points:
            weak_topics = [wp['topic'] for wp in weak_points[:3]]
            suggestions.append(f"⚠️ 薄弱知识点：{', '.join(weak_topics)}，建议重点复习")
        
        # 基于掌握度建议
        if stats['avg_mastery'] < 50:
            suggestions.append("💡 整体掌握度较低，建议系统性地从基础开始学习")
        elif stats['avg_mastery'] > 80:
            suggestions.append("🌟 学习效果良好！可以尝试更深入的内容或帮助其他同学")
        
        if not suggestions:
            suggestions.append("✨ 继续保持当前的学习节奏，你做得很好！")
        
        return suggestions
    
    def analyze_question_topic(self, question: str) -> str:
        """
        分析问题所属主题
        基于关键词匹配进行简单分类
        """
        # 主题关键词映射
        topic_keywords = {
            "傅里叶": ["傅里叶", "fourier", "频谱", "频域"],
            "拉普拉斯": ["拉普拉斯", "laplace", "s域"],
            "卷积": ["卷积", "convolution"],
            "信号与系统": ["信号", "系统", "冲激", "阶跃"],
            "微积分": ["微分", "积分", "导数", "极限"],
            "线性代数": ["矩阵", "向量", "特征值", "行列式"],
            "概率统计": ["概率", "期望", "方差", "分布"],
            "电路": ["电路", "电压", "电流", "电阻"],
            "数学物理方程": ["偏微分", "波动方程", "热传导", "达朗贝尔"],
            "复变函数": ["复变", "解析", "留数", "积分"]
        }
        
        question_lower = question.lower()
        
        for topic, keywords in topic_keywords.items():
            for keyword in keywords:
                if keyword.lower() in question_lower:
                    return topic
        
        return "综合问题"


def extract_topic_from_question(question: str) -> str:
    """从问题中提取主题的便捷函数"""
    analyzer = LearningAnalytics(0)  # user_id 不影响主题提取
    return analyzer.analyze_question_topic(question)
