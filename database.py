"""
数据库模块 - 学生画像系统数据存储
支持用户信息、问答历史、知识点掌握度、测验结果、学习时长
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

# 数据库文件路径
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "students.db")


def ensure_db_dir():
    """确保数据库目录存在"""
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)


@contextmanager
def get_db_connection():
    """获取数据库连接的上下文管理器"""
    ensure_db_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 返回字典格式
    try:
        yield conn
    finally:
        conn.close()


def init_database():
    """初始化数据库，创建所有必要的表"""
    ensure_db_dir()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # 用户表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                level TEXT DEFAULT '本科生',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 问答历史表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                topic TEXT DEFAULT '未分类',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # 知识点掌握度表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                topic TEXT NOT NULL,
                mastery_score REAL DEFAULT 0.0,
                question_count INTEGER DEFAULT 0,
                correct_count INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(user_id, topic)
            )
        ''')
        
        # 测验结果表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quiz_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                question TEXT NOT NULL,
                user_answer TEXT,
                correct_answer TEXT,
                is_correct INTEGER DEFAULT 0,
                topic TEXT DEFAULT '未分类',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # 学习时长表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS study_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                duration_minutes REAL DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        conn.commit()
        print("✅ 数据库初始化完成")


# ==================== 用户相关操作 ====================

def create_user(username: str, password_hash: str, level: str = "本科生") -> Optional[int]:
    """创建新用户，返回用户ID"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO users (username, password_hash, level) VALUES (?, ?, ?)",
                (username, password_hash, level)
            )
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            return None  # 用户名已存在


def get_user_by_username(username: str) -> Optional[Dict]:
    """通过用户名获取用户信息"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def get_user_by_id(user_id: int) -> Optional[Dict]:
    """通过ID获取用户信息"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def update_user_level(user_id: int, level: str) -> bool:
    """更新用户学习级别"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET level = ? WHERE id = ?",
            (level, user_id)
        )
        conn.commit()
        return cursor.rowcount > 0


# ==================== 问答历史相关操作 ====================

def save_chat_history(user_id: int, question: str, answer: str, topic: str = "未分类") -> int:
    """保存问答记录"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_history (user_id, question, answer, topic) VALUES (?, ?, ?, ?)",
            (user_id, question, answer, topic)
        )
        conn.commit()
        return cursor.lastrowid


def get_chat_history(user_id: int, limit: int = 50) -> List[Dict]:
    """获取用户的问答历史"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM chat_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
            (user_id, limit)
        )
        return [dict(row) for row in cursor.fetchall()]


def get_chat_count_by_topic(user_id: int) -> Dict[str, int]:
    """统计用户各主题的问答数量"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT topic, COUNT(*) as count FROM chat_history WHERE user_id = ? GROUP BY topic",
            (user_id,)
        )
        return {row['topic']: row['count'] for row in cursor.fetchall()}


def get_total_questions(user_id: int) -> int:
    """获取用户总问题数"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) as count FROM chat_history WHERE user_id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        return row['count'] if row else 0


# ==================== 知识点掌握度相关操作 ====================

def update_knowledge_point(user_id: int, topic: str, is_correct: bool = True) -> None:
    """更新知识点掌握度"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # 检查是否已存在该知识点记录
        cursor.execute(
            "SELECT * FROM knowledge_points WHERE user_id = ? AND topic = ?",
            (user_id, topic)
        )
        existing = cursor.fetchone()
        
        if existing:
            # 更新现有记录
            new_question_count = existing['question_count'] + 1
            new_correct_count = existing['correct_count'] + (1 if is_correct else 0)
            new_mastery = new_correct_count / new_question_count
            
            cursor.execute('''
                UPDATE knowledge_points 
                SET question_count = ?, correct_count = ?, mastery_score = ?, last_updated = ?
                WHERE user_id = ? AND topic = ?
            ''', (new_question_count, new_correct_count, new_mastery, datetime.now(), user_id, topic))
        else:
            # 创建新记录
            mastery = 1.0 if is_correct else 0.0
            cursor.execute('''
                INSERT INTO knowledge_points (user_id, topic, mastery_score, question_count, correct_count)
                VALUES (?, ?, ?, 1, ?)
            ''', (user_id, topic, mastery, 1 if is_correct else 0))
        
        conn.commit()


def get_knowledge_points(user_id: int) -> List[Dict]:
    """获取用户所有知识点掌握情况"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM knowledge_points WHERE user_id = ? ORDER BY mastery_score ASC",
            (user_id,)
        )
        return [dict(row) for row in cursor.fetchall()]


def get_weak_points(user_id: int, threshold: float = 0.6) -> List[Dict]:
    """获取用户薄弱知识点（掌握度低于阈值）"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM knowledge_points WHERE user_id = ? AND mastery_score < ? ORDER BY mastery_score ASC",
            (user_id, threshold)
        )
        return [dict(row) for row in cursor.fetchall()]


# ==================== 测验结果相关操作 ====================

def save_quiz_result(user_id: int, question: str, user_answer: str, 
                     correct_answer: str, is_correct: bool, topic: str = "未分类") -> int:
    """保存测验结果"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO quiz_results (user_id, question, user_answer, correct_answer, is_correct, topic)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, question, user_answer, correct_answer, 1 if is_correct else 0, topic))
        conn.commit()
        
        # 同时更新知识点掌握度
        update_knowledge_point(user_id, topic, is_correct)
        
        return cursor.lastrowid


def get_quiz_results(user_id: int, limit: int = 100) -> List[Dict]:
    """获取用户测验历史"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM quiz_results WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
            (user_id, limit)
        )
        return [dict(row) for row in cursor.fetchall()]


def get_quiz_accuracy(user_id: int) -> float:
    """获取用户总体正确率"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) as total, SUM(is_correct) as correct FROM quiz_results WHERE user_id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        if row and row['total'] > 0:
            return row['correct'] / row['total']
        return 0.0


def get_quiz_accuracy_by_topic(user_id: int) -> Dict[str, float]:
    """获取用户各主题正确率"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT topic, COUNT(*) as total, SUM(is_correct) as correct 
            FROM quiz_results WHERE user_id = ? GROUP BY topic
        ''', (user_id,))
        result = {}
        for row in cursor.fetchall():
            if row['total'] > 0:
                result[row['topic']] = row['correct'] / row['total']
        return result


# ==================== 学习时长相关操作 ====================

def start_study_session(user_id: int) -> int:
    """开始学习会话，返回会话ID"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO study_sessions (user_id, start_time) VALUES (?, ?)",
            (user_id, datetime.now())
        )
        conn.commit()
        return cursor.lastrowid


def end_study_session(session_id: int) -> None:
    """结束学习会话"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        end_time = datetime.now()
        
        # 获取开始时间计算时长
        cursor.execute("SELECT start_time FROM study_sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        if row:
            start_time = datetime.fromisoformat(row['start_time'])
            duration = (end_time - start_time).total_seconds() / 60  # 转换为分钟
            
            cursor.execute('''
                UPDATE study_sessions SET end_time = ?, duration_minutes = ? WHERE id = ?
            ''', (end_time, duration, session_id))
            conn.commit()


def get_total_study_time(user_id: int) -> float:
    """获取用户总学习时长（分钟）"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT SUM(duration_minutes) as total FROM study_sessions WHERE user_id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        return row['total'] if row and row['total'] else 0.0


def get_daily_study_time(user_id: int, days: int = 30) -> List[Dict]:
    """获取用户每日学习时长（最近N天）"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT DATE(start_time) as date, SUM(duration_minutes) as minutes
            FROM study_sessions 
            WHERE user_id = ? AND start_time >= DATE('now', ?)
            GROUP BY DATE(start_time)
            ORDER BY date ASC
        ''', (user_id, f'-{days} days'))
        return [dict(row) for row in cursor.fetchall()]


def get_study_sessions(user_id: int, limit: int = 50) -> List[Dict]:
    """获取用户学习会话历史"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM study_sessions WHERE user_id = ? ORDER BY start_time DESC LIMIT ?",
            (user_id, limit)
        )
        return [dict(row) for row in cursor.fetchall()]


# ==================== 统计汇总 ====================

# 记录测验结果

def save_quiz_record(user_id, topic, score_count, mastery_status):
    """
    记录测验结果
    :param score_count: 答对题数 (0-3)
    :param mastery_status: 掌握状态字符串 (已掌握/掌握75%/未掌握)
    """
    conn = sqlite3.connect('student_profile.db')
    c = conn.cursor()
    
    # 确保存储表存在
    c.execute('''CREATE TABLE IF NOT EXISTS quiz_records
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id TEXT,
                  topic TEXT,
                  score INTEGER,
                  mastery_level TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    c.execute("INSERT INTO quiz_records (user_id, topic, score, mastery_level) VALUES (?, ?, ?, ?)",
              (user_id, topic, score_count, mastery_status))
    
    conn.commit()
    conn.close()

def get_user_statistics(user_id: int) -> Dict[str, Any]:
    """获取用户学习统计汇总"""
    return {
        "total_questions": get_total_questions(user_id),
        "total_study_minutes": get_total_study_time(user_id),
        "quiz_accuracy": get_quiz_accuracy(user_id),
        "knowledge_points": get_knowledge_points(user_id),
        "weak_points": get_weak_points(user_id),
        "daily_study_time": get_daily_study_time(user_id),
        "topic_distribution": get_chat_count_by_topic(user_id),
        "topic_accuracy": get_quiz_accuracy_by_topic(user_id)
    }


# 初始化数据库
if __name__ == "__main__":
    init_database()
